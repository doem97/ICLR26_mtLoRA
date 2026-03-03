"""
Block-Level Adapters for mtLoRA

Implements parallel residual adapters that can be applied at the block level
(attention or FFN blocks) instead of component level (q_proj, v_proj, etc.)

Two styles:
1. LowRankBlockAdapter: down -> up (no activation, pure low-rank)
2. BottleneckBlockAdapter: down -> ReLU -> up (AdaptFormer style)

Note: Router architecture is aligned with component-level LoRA (simple Linear)
for consistency and fair comparison. Per-token routing is used instead of
mean-pooling to preserve sequence information.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankBlockAdapter(nn.Module):
    """
    Low-rank block-level adapter following HydraLoRA's asymmetric structure.

    Architecture: x -> A (shared) -> B_i (expert-specific) -> output
    No non-linearity between projections (pure low-rank decomposition).

    Args:
        dim: Hidden dimension (e.g., 4096 for LLaMA-7B)
        rank: Bottleneck dimension
        num_experts: Number of expert B matrices
        lora_alpha: Scaling factor alpha
        lora_dropout: Dropout probability
        enable_routing: Whether to use dynamic routing
        enable_blc: Whether to calculate balance loss coefficient
        enable_fine_grained_routing: Whether to use dimension-wise routing
        routing_group_size: Group size for fine-grained routing
    """

    def __init__(
        self,
        dim,
        rank,
        num_experts=1,
        lora_alpha=16,
        lora_dropout=0.0,
        enable_routing=True,
        enable_blc=False,
        enable_fine_grained_routing=False,
        routing_group_size=1,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.num_experts = num_experts
        self.enable_routing = enable_routing
        self.enable_blc = enable_blc
        self.enable_fine_grained_routing = enable_fine_grained_routing
        self.routing_group_size = routing_group_size

        # Scaling factor
        self.scaling = lora_alpha / rank

        # Dropout
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Shared down-projection (A matrix in HydraLoRA)
        self.lora_A = nn.Linear(dim, rank, bias=False)

        # Multiple up-projections (B matrices, one per expert)
        self.lora_B = nn.ModuleList([nn.Linear(rank, dim, bias=False) for _ in range(num_experts)])

        # Router for expert selection - using simple Linear for consistency with component-level LoRA
        if enable_routing and num_experts > 1:
            if enable_fine_grained_routing:
                # Fine-grained routing: output dimension = num_experts * num_groups
                num_groups = dim // routing_group_size
                router_output_dim = num_experts * num_groups
                self.num_groups = num_groups
            else:
                # Standard scalar routing
                router_output_dim = num_experts
                self.num_groups = 1

            # Simple Linear router (consistent with component-level LoRA in lora.py)
            self.lora_route = nn.Linear(dim, router_output_dim, bias=False)
        else:
            self.lora_route = None
            self.num_groups = 1

        # Initialize weights
        self._reset_parameters()

        # For BLC collection
        self.last_blc = None

    def _reset_parameters(self):
        """Initialize adapter parameters following LoRA conventions."""
        # A matrix: Kaiming uniform (like LoRA)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))

        # B matrices: zero initialization (so adapter starts as identity)
        for b_layer in self.lora_B:
            nn.init.zeros_(b_layer.weight)

        # Router: Kaiming uniform (consistent with component-level LoRA)
        if self.lora_route is not None:
            nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def cv_squared(self, x):
        """Calculate coefficient of variation squared for BLC."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, hydra_task_id=None, **kwargs):
        """
        Forward pass with optional routing.

        Uses per-token routing (consistent with component-level LoRA) instead of
        mean-pooling to preserve sequence information.

        Args:
            x: Input tensor [batch, seq_len, dim]
            hydra_task_id: Task ID for routing (optional)

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Dtype conversion for compatibility (consistent with component-level LoRA)
        x_for_lora = x.to(self.lora_A.weight.dtype) if hasattr(self.lora_A, "weight") else x

        # Down-projection (shared A matrix)
        hidden = self.lora_A(self.dropout(x_for_lora))

        if self.num_experts == 1:
            # Single expert: simple forward
            result = self.lora_B[0](hidden) * self.scaling
            self.last_blc = None
        else:
            # Multiple experts: per-token routing (consistent with component-level LoRA)
            # Get routing weights using the input directly (no mean pooling)
            x_for_route = x.to(self.lora_route.weight.dtype) if hasattr(self.lora_route, "weight") else x
            route_logits = self.lora_route(x_for_route)

            if self.enable_fine_grained_routing:
                # Fine-grained routing: dimension-wise weights
                # route_logits shape: [batch, seq_len, lora_num * num_groups]
                # Reshape to [batch, seq_len, num_groups, lora_num]
                batch_size = route_logits.shape[0]
                if len(route_logits.shape) == 3:
                    seq_len = route_logits.shape[1]
                    route_logits = route_logits.view(batch_size, seq_len, self.num_groups, self.num_experts)
                else:  # 2D tensor
                    route_logits = route_logits.view(batch_size, self.num_groups, self.num_experts)

                # Softmax over experts dimension
                route_weight = F.softmax(route_logits, dim=-1, dtype=torch.float32).to(x.dtype)

                # Initialize result
                result = torch.zeros_like(x)

                # Apply each expert with dimension-wise weights
                for i in range(self.num_experts):
                    expert_output = self.lora_B[i](hidden)

                    # Get dimension-wise weights for expert i
                    if len(route_weight.shape) == 4:
                        weight_i = route_weight[:, :, :, i]  # [batch, seq_len, num_groups]
                        # Repeat each weight for routing_group_size dimensions
                        weight_i = weight_i.repeat_interleave(self.routing_group_size, dim=-1)  # [batch, seq_len, dim]
                    else:  # 3D tensor
                        weight_i = route_weight[:, :, i]  # [batch, num_groups]
                        weight_i = weight_i.repeat_interleave(self.routing_group_size, dim=-1)  # [batch, dim]

                    # Element-wise multiplication with dimension-specific weights
                    result = result + (weight_i * expert_output * self.scaling)

                # Calculate BLC if enabled
                if self.enable_blc and self.training:
                    # For fine-grained routing, average over groups and sequence
                    if len(route_weight.shape) == 4:
                        blc = self.cv_squared(route_weight.mean(dim=(0, 1, 2)))
                    else:
                        blc = self.cv_squared(route_weight.mean(dim=(0, 1)))
                    self.last_blc = blc
                else:
                    self.last_blc = None

            else:
                # Standard scalar routing (per-token)
                route_weight = F.softmax(route_logits, dim=-1, dtype=torch.float32).to(x.dtype)

                # Initialize result
                result = torch.zeros_like(x)

                # Apply each expert with scalar weights
                for i in range(self.num_experts):
                    expert_output = self.lora_B[i](hidden)

                    # Handle different tensor dimensions (2D or 3D)
                    if len(route_weight.shape) == 3:
                        weight_i = route_weight[:, :, i].unsqueeze(-1)  # [batch, seq_len, 1]
                    else:  # 2D tensor
                        weight_i = route_weight[:, i].unsqueeze(-1)  # [batch, 1]

                    result = result + (weight_i * expert_output * self.scaling)

                # Calculate BLC if enabled
                if self.enable_blc and self.training:
                    if len(route_weight.shape) == 3:
                        blc = self.cv_squared(route_weight.mean(dim=(0, 1)))
                    else:
                        blc = self.cv_squared(route_weight.mean(dim=0))
                    self.last_blc = blc
                else:
                    self.last_blc = None

        return result


class BottleneckBlockAdapter(nn.Module):
    """
    Bottleneck block-level adapter (AdaptFormer style) with ReLU activation.

    Architecture: x -> A (shared) -> ReLU -> B_i (expert-specific) -> output
    Includes non-linearity between projections.

    Args:
        dim: Hidden dimension
        rank: Bottleneck dimension
        num_experts: Number of expert B matrices
        lora_alpha: Scaling factor alpha
        lora_dropout: Dropout probability
        enable_routing: Whether to use dynamic routing
        enable_blc: Whether to calculate balance loss coefficient
        enable_fine_grained_routing: Whether to use dimension-wise routing
        routing_group_size: Group size for fine-grained routing
    """

    def __init__(
        self,
        dim,
        rank,
        num_experts=1,
        lora_alpha=16,
        lora_dropout=0.0,
        enable_routing=True,
        enable_blc=False,
        enable_fine_grained_routing=False,
        routing_group_size=1,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.num_experts = num_experts
        self.enable_routing = enable_routing
        self.enable_blc = enable_blc
        self.enable_fine_grained_routing = enable_fine_grained_routing
        self.routing_group_size = routing_group_size

        # Scaling factor
        self.scaling = lora_alpha / rank

        # Dropout
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Shared down-projection with ReLU activation
        self.lora_A = nn.Linear(dim, rank, bias=False)
        self.activation = nn.ReLU()

        # Multiple up-projections (one per expert)
        self.lora_B = nn.ModuleList([nn.Linear(rank, dim, bias=False) for _ in range(num_experts)])

        # Router for expert selection - using simple Linear for consistency with component-level LoRA
        if enable_routing and num_experts > 1:
            if enable_fine_grained_routing:
                num_groups = dim // routing_group_size
                router_output_dim = num_experts * num_groups
                self.num_groups = num_groups
            else:
                router_output_dim = num_experts
                self.num_groups = 1

            # Simple Linear router (consistent with component-level LoRA)
            self.lora_route = nn.Linear(dim, router_output_dim, bias=False)
        else:
            self.lora_route = None
            self.num_groups = 1

        # Initialize weights
        self._reset_parameters()

        # For BLC collection
        self.last_blc = None

    def _reset_parameters(self):
        """Initialize adapter parameters."""
        # A matrix: Kaiming uniform for ReLU
        nn.init.kaiming_uniform_(self.lora_A.weight, nonlinearity="relu")

        # B matrices: small random init (not zero, since we have ReLU)
        for b_layer in self.lora_B:
            nn.init.normal_(b_layer.weight, std=0.02)

        # Router: Kaiming uniform (consistent with component-level LoRA)
        if self.lora_route is not None:
            nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def cv_squared(self, x):
        """Calculate coefficient of variation squared for BLC."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, hydra_task_id=None, **kwargs):
        """
        Forward pass with ReLU activation and optional routing.

        Uses per-token routing (consistent with component-level LoRA) instead of
        mean-pooling to preserve sequence information.

        Args:
            x: Input tensor [batch, seq_len, dim]
            hydra_task_id: Task ID for routing (optional)

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Dtype conversion for compatibility
        x_for_lora = x.to(self.lora_A.weight.dtype) if hasattr(self.lora_A, "weight") else x

        # Down-projection with activation (bottleneck)
        hidden = self.activation(self.lora_A(self.dropout(x_for_lora)))

        if self.num_experts == 1:
            # Single expert
            result = self.lora_B[0](hidden) * self.scaling
            self.last_blc = None
        else:
            # Multiple experts with per-token routing (consistent with component-level LoRA)
            x_for_route = x.to(self.lora_route.weight.dtype) if hasattr(self.lora_route, "weight") else x
            route_logits = self.lora_route(x_for_route)

            if self.enable_fine_grained_routing:
                # Fine-grained routing: dimension-wise weights
                batch_size = route_logits.shape[0]
                if len(route_logits.shape) == 3:
                    seq_len = route_logits.shape[1]
                    route_logits = route_logits.view(batch_size, seq_len, self.num_groups, self.num_experts)
                else:
                    route_logits = route_logits.view(batch_size, self.num_groups, self.num_experts)

                route_weight = F.softmax(route_logits, dim=-1, dtype=torch.float32).to(x.dtype)

                result = torch.zeros_like(x)
                for i in range(self.num_experts):
                    expert_output = self.lora_B[i](hidden)
                    if len(route_weight.shape) == 4:
                        weight_i = route_weight[:, :, :, i]
                        weight_i = weight_i.repeat_interleave(self.routing_group_size, dim=-1)
                    else:
                        weight_i = route_weight[:, :, i]
                        weight_i = weight_i.repeat_interleave(self.routing_group_size, dim=-1)
                    result = result + (weight_i * expert_output * self.scaling)

                if self.enable_blc and self.training:
                    if len(route_weight.shape) == 4:
                        blc = self.cv_squared(route_weight.mean(dim=(0, 1, 2)))
                    else:
                        blc = self.cv_squared(route_weight.mean(dim=(0, 1)))
                    self.last_blc = blc
                else:
                    self.last_blc = None
            else:
                # Standard scalar routing (per-token)
                route_weight = F.softmax(route_logits, dim=-1, dtype=torch.float32).to(x.dtype)

                result = torch.zeros_like(x)
                for i in range(self.num_experts):
                    expert_output = self.lora_B[i](hidden)
                    if len(route_weight.shape) == 3:
                        weight_i = route_weight[:, :, i].unsqueeze(-1)
                    else:
                        weight_i = route_weight[:, i].unsqueeze(-1)
                    result = result + (weight_i * expert_output * self.scaling)

                if self.enable_blc and self.training:
                    if len(route_weight.shape) == 3:
                        blc = self.cv_squared(route_weight.mean(dim=(0, 1)))
                    else:
                        blc = self.cv_squared(route_weight.mean(dim=0))
                    self.last_blc = blc
                else:
                    self.last_blc = None

        return result


class MoEAdaptFormerBlockAdapter(nn.Module):
    """
    MoE of AdaptFormers - Multiple AdaptFormer experts with routing.

    Key differences from BottleneckBlockAdapter:
    1. Zero initialization for B matrices (gradual introduction like AdaptFormer)
    2. Per-expert learnable scale parameters (instead of fixed alpha/rank)
    3. ReLU activation between down and up projections

    Architecture: x -> A (shared) -> ReLU -> s_i * B_i (expert-specific) -> output

    This follows the AdaptFormer design philosophy:
    - Initial output is zero (preserves pretrained behavior)
    - Adapter contribution grows gradually during training
    - Each expert has its own learnable magnitude

    Args:
        dim: Hidden dimension (e.g., 4096 for LLaMA-7B)
        rank: Bottleneck dimension
        num_experts: Number of expert B matrices
        lora_dropout: Dropout probability
        enable_routing: Whether to use dynamic routing
        enable_blc: Whether to calculate balance loss coefficient
        enable_fine_grained_routing: Whether to use dimension-wise routing
        routing_group_size: Group size for fine-grained routing
        init_scale: Initial value for learnable scales (default 0.0 for zero-init)
    """

    def __init__(
        self,
        dim,
        rank,
        num_experts=1,
        lora_dropout=0.0,
        enable_routing=True,
        enable_blc=False,
        enable_fine_grained_routing=False,
        routing_group_size=1,
        init_scale=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.num_experts = num_experts
        self.enable_routing = enable_routing
        self.enable_blc = enable_blc
        self.enable_fine_grained_routing = enable_fine_grained_routing
        self.routing_group_size = routing_group_size

        # Dropout
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Shared down-projection
        self.lora_A = nn.Linear(dim, rank, bias=False)

        # ReLU activation (AdaptFormer style)
        self.activation = nn.ReLU()

        # Multiple up-projections (B matrices, one per expert)
        self.lora_B = nn.ModuleList([nn.Linear(rank, dim, bias=False) for _ in range(num_experts)])

        # Per-expert learnable scale parameters (AdaptFormer style)
        # Initialized to init_scale (default 0) for zero-init behavior
        self.expert_scales = nn.ParameterList([nn.Parameter(torch.full((1,), init_scale)) for _ in range(num_experts)])

        # Router for expert selection
        if enable_routing and num_experts > 1:
            if enable_fine_grained_routing:
                num_groups = dim // routing_group_size
                router_output_dim = num_experts * num_groups
                self.num_groups = num_groups
            else:
                router_output_dim = num_experts
                self.num_groups = 1

            self.lora_route = nn.Linear(dim, router_output_dim, bias=False)
        else:
            self.lora_route = None
            self.num_groups = 1

        # Initialize weights
        self._reset_parameters()

        # For BLC collection
        self.last_blc = None

    def _reset_parameters(self):
        """Initialize adapter parameters following AdaptFormer conventions."""
        # A matrix: Kaiming uniform for ReLU
        nn.init.kaiming_uniform_(self.lora_A.weight, nonlinearity="relu")

        # B matrices: ZERO initialization (AdaptFormer style)
        # Combined with zero-init scales, this ensures initial output is zero
        for b_layer in self.lora_B:
            nn.init.zeros_(b_layer.weight)

        # Router: Kaiming uniform
        if self.lora_route is not None:
            nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

        # Note: expert_scales are initialized in __init__ to init_scale (default 0)

    def cv_squared(self, x):
        """Calculate coefficient of variation squared for BLC."""
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x, hydra_task_id=None, **kwargs):
        """
        Forward pass with ReLU activation, per-expert scaling, and optional routing.

        Args:
            x: Input tensor [batch, seq_len, dim]
            hydra_task_id: Task ID for routing (optional)

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # Dtype conversion for compatibility
        x_for_lora = x.to(self.lora_A.weight.dtype) if hasattr(self.lora_A, "weight") else x

        # Down-projection with activation (bottleneck)
        hidden = self.activation(self.lora_A(self.dropout(x_for_lora)))

        if self.num_experts == 1:
            # Single expert: apply learnable scale
            result = self.expert_scales[0] * self.lora_B[0](hidden)
            self.last_blc = None
        else:
            # Multiple experts with per-token routing
            x_for_route = x.to(self.lora_route.weight.dtype) if hasattr(self.lora_route, "weight") else x
            route_logits = self.lora_route(x_for_route)

            if self.enable_fine_grained_routing:
                # Fine-grained routing: dimension-wise weights
                batch_size = route_logits.shape[0]
                if len(route_logits.shape) == 3:
                    seq_len = route_logits.shape[1]
                    route_logits = route_logits.view(batch_size, seq_len, self.num_groups, self.num_experts)
                else:
                    route_logits = route_logits.view(batch_size, self.num_groups, self.num_experts)

                route_weight = F.softmax(route_logits, dim=-1, dtype=torch.float32).to(x.dtype)

                result = torch.zeros_like(x)
                for i in range(self.num_experts):
                    # Apply per-expert learnable scale
                    expert_output = self.expert_scales[i] * self.lora_B[i](hidden)

                    if len(route_weight.shape) == 4:
                        weight_i = route_weight[:, :, :, i]
                        weight_i = weight_i.repeat_interleave(self.routing_group_size, dim=-1)
                    else:
                        weight_i = route_weight[:, :, i]
                        weight_i = weight_i.repeat_interleave(self.routing_group_size, dim=-1)

                    result = result + (weight_i * expert_output)

                if self.enable_blc and self.training:
                    if len(route_weight.shape) == 4:
                        blc = self.cv_squared(route_weight.mean(dim=(0, 1, 2)))
                    else:
                        blc = self.cv_squared(route_weight.mean(dim=(0, 1)))
                    self.last_blc = blc
                else:
                    self.last_blc = None
            else:
                # Standard scalar routing (per-token)
                route_weight = F.softmax(route_logits, dim=-1, dtype=torch.float32).to(x.dtype)

                result = torch.zeros_like(x)
                for i in range(self.num_experts):
                    # Apply per-expert learnable scale
                    expert_output = self.expert_scales[i] * self.lora_B[i](hidden)

                    if len(route_weight.shape) == 3:
                        weight_i = route_weight[:, :, i].unsqueeze(-1)
                    else:
                        weight_i = route_weight[:, i].unsqueeze(-1)

                    result = result + (weight_i * expert_output)

                if self.enable_blc and self.training:
                    if len(route_weight.shape) == 3:
                        blc = self.cv_squared(route_weight.mean(dim=(0, 1)))
                    else:
                        blc = self.cv_squared(route_weight.mean(dim=0))
                    self.last_blc = blc
                else:
                    self.last_blc = None

        return result
