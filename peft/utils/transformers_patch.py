"""
Monkey patches for HuggingFace Transformers to support HydraLoRA.

This module patches the Llama model classes to support:
1. Task-aware routing (passing task IDs through the forward pass)
2. Balance loss coefficient (BLC) collection from LoRA expert routing

These modifications enable HydraLoRA to route different tasks to different LoRA experts
and collect balance loss to ensure expert load balancing.

Why monkey patching instead of vendoring?
- Only 3 methods need modification in a 2000+ file library
- Keeps codebase lightweight (~100 lines vs 413MB)
- Easier to maintain and sync with upstream transformers updates
- Standard transformers can be used for other models

Usage:
    from peft.utils.transformers_patch import patch_llama_for_hydralora
    patch_llama_for_hydralora()
"""

import torch

# Python 3.10 compatibility - Unpack was added to typing in 3.11
try:
    from typing import Unpack
except ImportError:
    pass

# TransformersKwargs may not be available in all versions
try:
    from transformers.utils import TransformersKwargs
except ImportError:
    # Fallback: use TypedDict or just rely on **kwargs without type annotation
    TransformersKwargs = dict  # Simple fallback


def patch_llama_for_hydralora():
    """
    Patches transformers.models.llama to support HydraLoRA's task routing and BLC collection.

    This function should be called once at the beginning of training/evaluation scripts,
    before any model loading occurs.

    Modified methods:
    - LlamaMLP.forward(): Adds hydra_task_id parameter and passes it to LoRA layers
    - LlamaDecoderLayer.forward(): Passes hydra_task_id through to MLP
    - LlamaModel.forward(): Passes hydra_task_id through decoder layers
    - LlamaForCausalLM.forward(): Accepts task_types and passes as hydra_task_id to model
    """
    try:
        from transformers.models.llama.modeling_llama import (
            LlamaAttention,
            LlamaDecoderLayer,
            LlamaForCausalLM,
            LlamaMLP,
            LlamaModel,
        )
    except ImportError as e:
        raise ImportError(
            "Failed to import Llama classes from transformers. "
            "Please ensure transformers is installed: pip install transformers"
        ) from e

    # Import attention variants (different transformers versions may have different implementations)
    attention_classes = [LlamaAttention]
    try:
        from transformers.models.llama.modeling_llama import LlamaSdpaAttention

        attention_classes.append(LlamaSdpaAttention)
    except ImportError:
        LlamaSdpaAttention = None

    try:
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2

        attention_classes.append(LlamaFlashAttention2)
    except ImportError:
        LlamaFlashAttention2 = None

    # Store original methods for reference
    _original_mlp_forward = LlamaMLP.forward
    _original_attention_forward = LlamaAttention.forward
    _original_decoder_forward = LlamaDecoderLayer.forward
    _original_model_forward = LlamaModel.forward
    _original_causal_lm_forward = LlamaForCausalLM.forward

    # Store original forwards for all attention variants
    _original_attention_forwards = {LlamaAttention: LlamaAttention.forward}
    if LlamaSdpaAttention:
        _original_attention_forwards[LlamaSdpaAttention] = LlamaSdpaAttention.forward
    if LlamaFlashAttention2:
        _original_attention_forwards[LlamaFlashAttention2] = LlamaFlashAttention2.forward

    # Patch LlamaMLP.forward() to support task routing
    def patched_mlp_forward(self, x, hydra_task_id=None):
        """
        Modified MLP forward pass that supports task-aware routing.

        Args:
            x: Input tensor
            hydra_task_id: Task identifier for routing to appropriate LoRA experts

        Returns:
            Output tensor (BLC is collected via self.last_blc attribute if needed)
        """
        # Optimization: If no task routing needed (e.g. AdaLoRA, Standard LoRA),
        # use standard forward path to avoid overhead and potential errors
        if hydra_task_id is None:
            gate_output = self.gate_proj(x)
            up_output = self.up_proj(x)
            activated_intermediate = self.act_fn(gate_output) * up_output
            down_output = self.down_proj(activated_intermediate)
            return down_output

        # When LoRA layers are present and task routing is requested
        try:
            gate_output = self.gate_proj(x, task_types=hydra_task_id)
            up_output = self.up_proj(x, task_types=hydra_task_id)
            activated_intermediate = self.act_fn(gate_output) * up_output
            down_output = self.down_proj(activated_intermediate, task_types=hydra_task_id)
        except TypeError:
            # Fallback for layers that don't support task_types even if requested
            gate_output = self.gate_proj(x)
            up_output = self.up_proj(x)
            activated_intermediate = self.act_fn(gate_output) * up_output
            down_output = self.down_proj(activated_intermediate)

        return down_output

    # Patch LlamaAttention.forward() to pass kwargs to projection layers
    # This wrapper intercepts calls to projection layers and adds kwargs
    def patched_attention_forward(self, hidden_states, *args, **kwargs):
        """
        Wrapper for LlamaAttention that extracts hydra_task_id from kwargs
        and passes it to all projection layers (q/k/v/o).

        Note: This is specifically needed for MMOELoRA which requires task_id
        in attention projections. HydraLoRA does per-sample routing and doesn't
        need task_id in attention layers.

        Works with all attention variants (LlamaAttention, LlamaSdpaAttention, LlamaFlashAttention2).
        """
        # Extract hydra_task_id if present
        hydra_task_id = kwargs.pop("hydra_task_id", None)

        # Get the correct original forward for this instance's class
        original_forward = _original_attention_forwards.get(type(self), _original_attention_forward)

        # Optimization: If no task routing needed (e.g. AdaLoRA, Standard LoRA),
        # skip all patching logic and call original forward directly.
        # This avoids memory leaks from dynamic closure creation in Autograd.
        if hydra_task_id is None:
            return original_forward(self, hidden_states, *args, **kwargs)

        # Call the original forward, but wrap projection calls
        # Store original forward methods
        orig_q = self.q_proj.forward
        orig_k = self.k_proj.forward
        orig_v = self.v_proj.forward
        orig_o = self.o_proj.forward

        # Create wrapped versions that pass hydra_task_id to MMOELoRA layers
        def wrap_proj(orig_fn):
            def wrapped(x):
                try:
                    # Pass hydra_task_id for MMOELoRA (HydraLoRA doesn't need it in attention)
                    return orig_fn(x, hydra_task_id=hydra_task_id)
                except TypeError:
                    # Fallback to original call for non-LoRA layers
                    return orig_fn(x)

            return wrapped

        # Temporarily replace forward methods
        self.q_proj.forward = wrap_proj(orig_q)
        self.k_proj.forward = wrap_proj(orig_k)
        self.v_proj.forward = wrap_proj(orig_v)
        self.o_proj.forward = wrap_proj(orig_o)

        try:
            result = original_forward(self, hidden_states, *args, **kwargs)
        finally:
            # Restore original forward methods
            self.q_proj.forward = orig_q
            self.k_proj.forward = orig_k
            self.v_proj.forward = orig_v
            self.o_proj.forward = orig_o

        return result

    # Patch LlamaDecoderLayer.forward() to pass task_id through (v4.36.2 compatible)
    def patched_decoder_forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,  # v4.36.2 uses singular past_key_value
        output_attentions=False,
        use_cache=False,
        hydra_task_id=None,  # Added for HydraLoRA
        **kwargs,
    ):
        """
        Modified decoder layer forward pass that propagates task_id to MLP.
        Compatible with transformers 4.36.2 API.

        Args:
            hydra_task_id: Task identifier passed through to MLP for expert routing
            (other args same as original)
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention - pass task_id through kwargs for LoRA layers
        # Add hydra_task_id to kwargs so it's passed to projection layers (q/k/v/o)
        attn_kwargs = {**kwargs}
        if hydra_task_id is not None:
            attn_kwargs["hydra_task_id"] = hydra_task_id

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,  # Singular in v4.36.2
            output_attentions=output_attentions,
            use_cache=use_cache,
            **attn_kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected (MLP) - pass task_id for expert routing
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, hydra_task_id=hydra_task_id)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

    # Patch LlamaModel.forward() to accept and propagate task_id (v4.36.2 compatible)
    def patched_model_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        hydra_task_id=None,  # Added for HydraLoRA
    ):
        """
        Modified model forward pass that propagates task_id through all decoder layers.
        Compatible with transformers 4.36.2 API.

        Args:
            hydra_task_id: Task identifier for routing to appropriate LoRA experts
            (other args same as original)
        """
        from transformers.cache_utils import Cache, DynamicCache
        from transformers.modeling_attn_mask_utils import (
            _prepare_4d_causal_attention_mask,
            _prepare_4d_causal_attention_mask_for_sdpa,
        )
        from transformers.modeling_outputs import BaseModelOutputWithPast
        from transformers.utils import logging

        logger = logging.get_logger(__name__)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._use_sdpa and not output_attentions:
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                # Note: gradient checkpointing doesn't support hydra_task_id parameter passing
                # This is a limitation when using gradient checkpointing
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    hydra_task_id,  # Pass task_id for expert routing even in checkpointing
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    hydra_task_id=hydra_task_id,  # Pass task_id for expert routing
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Patch LlamaForCausalLM.forward() to accept task_types and forward to model
    def patched_causal_lm_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_types=None,  # Added for HydraLoRA - will be passed as hydra_task_id
        **kwargs,
    ):
        """
        Modified CausalLM forward pass that accepts task_types and forwards to model.

        This wraps the original logic but adds hydra_task_id parameter support for
        passing to the patched LlamaModel.

        Args:
            task_types: Task identifiers for routing (passed as hydra_task_id to LlamaModel)
            (other args same as original)
        """
        import torch.nn.functional as F
        from torch.nn import CrossEntropyLoss
        from transformers.modeling_outputs import CausalLMOutputWithPast

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Call patched model with hydra_task_id
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            hydra_task_id=task_types,  # Pass to patched LlamaModel
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # Apply the patches
    LlamaMLP.forward = patched_mlp_forward

    # Apply attention patch to all attention variants
    for attention_class in attention_classes:
        attention_class.forward = patched_attention_forward

    LlamaDecoderLayer.forward = patched_decoder_forward
    LlamaModel.forward = patched_model_forward
    LlamaForCausalLM.forward = patched_causal_lm_forward

    print("[OK] HydraLoRA transformers patches applied successfully")
    print("  - LlamaMLP.forward() now supports task routing")
    print(f"  - Attention.forward() passes task_id to projection layers ({len(attention_classes)} variants patched)")
    print("  - LlamaDecoderLayer.forward() propagates task_id")
    print("  - LlamaModel.forward() propagates task_id through layers")
    print("  - LlamaForCausalLM.forward() accepts task_types and forwards to model")


def inject_block_adapters(model, adapter_config):
    """
    Inject block-level adapters into LLaMA model.

    This function adds parallel adapter paths to attention and/or FFN blocks.
    The adapters are applied as: output = Block(x) + Adapter(x)

    Args:
        model: The LLaMA model (or wrapped model with base_model)
        adapter_config: Dict with adapter configuration:
            - enable_block_adapter: bool
            - block_adapter_type: "attention", "ffn", or "both"
            - block_adapter_style: "lowrank", "bottleneck", or "adaptformer"
            - block_adapter_rank: int
            - lora_nums: int (number of experts)
            - lora_alpha: float (not used for adaptformer style)
            - lora_dropout: float
            - enable_blc: bool
            - enable_fine_grained_routing: bool
            - routing_group_size: int
            - adaptformer_init_scale: float (only for adaptformer style, default 0.0)
    """
    if not adapter_config.get("enable_block_adapter", False):
        return

    from peft.tuners.block_adapters import (
        BottleneckBlockAdapter,
        LowRankBlockAdapter,
        MoEAdaptFormerBlockAdapter,
    )

    # Unwrap model if needed
    base_model = model
    if hasattr(model, "base_model"):
        if hasattr(model.base_model, "model"):
            base_model = model.base_model.model
        else:
            base_model = model.base_model
    if hasattr(base_model, "model"):
        base_model = base_model.model

    # Get configuration
    adapter_type = adapter_config.get("block_adapter_type", "attention")
    adapter_style = adapter_config.get("block_adapter_style", "lowrank")
    rank = adapter_config.get("block_adapter_rank", 16)
    num_experts = adapter_config.get("lora_nums", 1)
    lora_alpha = adapter_config.get("lora_alpha", 16)
    lora_dropout = adapter_config.get("lora_dropout", 0.0)
    enable_blc = adapter_config.get("enable_blc", False)
    enable_fine_grained_routing = adapter_config.get("enable_fine_grained_routing", False)
    routing_group_size = adapter_config.get("routing_group_size", 1)
    adaptformer_init_scale = adapter_config.get("adaptformer_init_scale", 0.0)

    # Select adapter class based on style
    if adapter_style == "lowrank":
        AdapterClass = LowRankBlockAdapter
    elif adapter_style == "bottleneck":
        AdapterClass = BottleneckBlockAdapter
    elif adapter_style == "adaptformer":
        AdapterClass = MoEAdaptFormerBlockAdapter
    else:
        raise ValueError(
            f"Unknown block_adapter_style: {adapter_style}. Must be 'lowrank', 'bottleneck', or 'adaptformer'"
        )

    # Get hidden dimension from model config
    hidden_size = base_model.config.hidden_size

    # Track injected adapters
    injected_count = 0

    # Build common kwargs for adapter creation
    def build_adapter_kwargs():
        """Build kwargs dict based on adapter style."""
        kwargs = {
            "dim": hidden_size,
            "rank": rank,
            "num_experts": num_experts,
            "lora_dropout": lora_dropout,
            "enable_routing": (num_experts > 1),
            "enable_blc": enable_blc,
            "enable_fine_grained_routing": enable_fine_grained_routing,
            "routing_group_size": routing_group_size,
        }
        # adaptformer uses learnable scales instead of lora_alpha
        if adapter_style != "adaptformer":
            kwargs["lora_alpha"] = lora_alpha
        else:
            kwargs["init_scale"] = adaptformer_init_scale
        return kwargs

    # Inject adapters into decoder layers
    for layer_idx, layer in enumerate(base_model.layers):
        # Inject into attention block
        if adapter_type in ["attention", "both"]:
            if not hasattr(layer.self_attn, "block_adapter"):
                adapter = AdapterClass(**build_adapter_kwargs())
                # Move to same device as layer
                adapter = adapter.to(layer.self_attn.q_proj.weight.device)
                adapter = adapter.to(layer.self_attn.q_proj.weight.dtype)
                layer.self_attn.block_adapter = adapter
                injected_count += 1

        # Inject into FFN/MLP block
        if adapter_type in ["ffn", "both"]:
            if not hasattr(layer.mlp, "block_adapter"):
                adapter = AdapterClass(**build_adapter_kwargs())
                # Move to same device as layer
                adapter = adapter.to(layer.mlp.gate_proj.weight.device)
                adapter = adapter.to(layer.mlp.gate_proj.weight.dtype)
                layer.mlp.block_adapter = adapter
                injected_count += 1

    print(f"[OK] Block-level adapters injected: {injected_count} adapters")
    print(f"  - Type: {adapter_type}, Style: {adapter_style}")
    print(f"  - Rank: {rank}, Experts: {num_experts}")
    if adapter_style == "adaptformer":
        print(f"  - AdaptFormer init_scale: {adaptformer_init_scale}")


def patch_llama_for_block_adapters():
    """
    Patch LLaMA attention and MLP forward methods to use block-level adapters.

    This modifies the forward passes to add parallel adapter outputs:
    - Attention: attn_out = self_attn(x) + adapter(x)
    - MLP: mlp_out = mlp(x) + adapter(x)

    Should be called after patch_llama_for_hydralora() and before injecting adapters.
    """
    try:
        from transformers.models.llama.modeling_llama import (
            LlamaAttention,
            LlamaDecoderLayer,
            LlamaMLP,
        )
    except ImportError as e:
        raise ImportError(
            "Failed to import Llama classes from transformers. "
            "Please ensure transformers is installed: pip install transformers"
        ) from e

    # Import attention variants
    attention_classes = [LlamaAttention]
    try:
        from transformers.models.llama.modeling_llama import LlamaSdpaAttention

        attention_classes.append(LlamaSdpaAttention)
    except ImportError:
        pass

    try:
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2

        attention_classes.append(LlamaFlashAttention2)
    except ImportError:
        pass

    # Store original forward methods
    _original_attention_forwards = {}
    for attn_class in attention_classes:
        _original_attention_forwards[attn_class] = attn_class.forward
    _original_mlp_forward = LlamaMLP.forward

    # Patch attention forward to add adapter output
    def patched_attention_with_adapter(self, hidden_states, *args, **kwargs):
        """Attention forward with optional block-level adapter."""
        # Get original forward for this class
        original_forward = _original_attention_forwards.get(self.__class__)
        if original_forward is None:
            # Fallback to first available
            original_forward = list(_original_attention_forwards.values())[0]

        # Original attention computation
        attn_outputs = original_forward(self, hidden_states, *args, **kwargs)

        # Add adapter if available
        if hasattr(self, "block_adapter") and self.block_adapter is not None:
            # attn_outputs is typically (attn_output, attn_weights, past_key_value)
            # We only modify the first element (attn_output)
            attn_output = attn_outputs[0]

            # Apply adapter to the INPUT (same as what attention received)
            adapter_output = self.block_adapter(hidden_states, **kwargs)

            # Add adapter output to attention output
            modified_attn_output = attn_output + adapter_output

            # Reconstruct outputs tuple
            attn_outputs = (modified_attn_output,) + attn_outputs[1:]

        return attn_outputs

    # Patch MLP forward to add adapter output
    def patched_mlp_with_adapter(self, x, hydra_task_id=None):
        """MLP forward with optional block-level adapter."""
        # Original MLP computation
        mlp_output = _original_mlp_forward(self, x, hydra_task_id=hydra_task_id)

        # Add adapter if available
        if hasattr(self, "block_adapter") and self.block_adapter is not None:
            # Apply adapter to the INPUT (same as what MLP received)
            adapter_kwargs = {}
            if hydra_task_id is not None:
                adapter_kwargs["hydra_task_id"] = hydra_task_id
            adapter_output = self.block_adapter(x, **adapter_kwargs)

            # Add adapter output to MLP output
            mlp_output = mlp_output + adapter_output

        return mlp_output

    # Apply patches
    for attn_class in attention_classes:
        attn_class.forward = patched_attention_with_adapter
    LlamaMLP.forward = patched_mlp_with_adapter

    print("[OK] Block-level adapter patches applied")
    print(f"  - Attention forward patched ({len(attention_classes)} variants)")
    print("  - MLP forward patched")
