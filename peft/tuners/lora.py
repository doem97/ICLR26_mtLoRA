# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose

logger = logging.getLogger(__name__)


@dataclass
class LoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: list[str] | str | None = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_nums: int = field(default=None, metadata={"help": "Numbers of Lora"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    merge_weights: bool = field(
        default=False,
        metadata={"help": "Merge weights of the original model and the Lora model"},
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: list[bool] | None = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(
        default="none",
        metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"},
    )
    modules_to_save: list[str] | None = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    enable_blc: bool = field(
        default=False,
        metadata={"help": "Enable Balance Loss Coefficient calculation. Default: False to avoid OOM issues."},
    )
    init_lora_weights: bool = field(
        default=True, metadata={"help": "Whether to initialize the LoRA weights. Default: True"}
    )
    # mtLoRA extensions
    enable_fine_grained_routing: bool = field(
        default=False, metadata={"help": "Enable fine-grained routing with dimension-wise weights"}
    )
    routing_group_size: int = field(
        default=1, metadata={"help": "Group size for fine-grained routing (1=full dimension-wise)"}
    )
    enable_spectral_reg: bool = field(
        default=False, metadata={"help": "Enable spectral-aware orthogonal regularization"}
    )
    spectral_reg_lambda: float = field(default=0.25, metadata={"help": "Weight for spectral regularization loss"})
    spectral_reg_frequency: int = field(
        default=1, metadata={"help": "Run SVD every N epochs for spectral regularization"}
    )
    spectral_reg_steps: int | None = field(default=None, metadata={"help": "If set, also run SVD every N steps"})
    # Block-level adaptation (mtLoRA third component)
    enable_block_adapter: bool = field(
        default=False, metadata={"help": "Enable block-level adaptation instead of component-level LoRA"}
    )
    block_adapter_type: str = field(
        default="attention", metadata={"help": "Where to apply block adapter: 'attention', 'ffn', or 'both'"}
    )
    block_adapter_style: str = field(default="lowrank", metadata={"help": "Adapter style: 'lowrank' or 'bottleneck'"})
    block_adapter_rank: int | None = field(
        default=None, metadata={"help": "Adapter rank (defaults to lora_rank if not set)"}
    )
    adaptformer_init_scale: float = field(
        default=0.0, metadata={"help": "Initial value for AdaptFormer per-expert learnable scales (default 0.0)"}
    )

    def __post_init__(self):
        self.peft_type = PeftType.LORA


class LoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):  # LoraConfig, CasualLM
        super().__init__()
        self.peft_config = config
        self.model = model

        # Handle block-level adaptation
        if getattr(config, "enable_block_adapter", False):
            # Warn if target_modules is specified but will be ignored
            if config.target_modules:
                logger.warning(
                    "Block adapter enabled - target_modules=%s will be ignored. "
                    "Block-level adaptation replaces component-level LoRA (q_proj, v_proj, etc.) "
                    "with adapters at the attention/FFN block level.",
                    config.target_modules,
                )

            # Import block adapter functions
            from peft.utils.transformers_patch import (
                inject_block_adapters,
                patch_llama_for_block_adapters,
            )

            # Apply patches to support block adapters
            patch_llama_for_block_adapters()

            # Inject adapters into model
            adapter_config = {
                "enable_block_adapter": config.enable_block_adapter,
                "block_adapter_type": config.block_adapter_type,
                "block_adapter_style": config.block_adapter_style,
                "block_adapter_rank": config.block_adapter_rank or config.r,
                "lora_nums": getattr(config, "lora_nums", 1),
                "lora_alpha": config.lora_alpha,
                "lora_dropout": config.lora_dropout,
                "enable_blc": getattr(config, "enable_blc", False),
                "enable_fine_grained_routing": getattr(config, "enable_fine_grained_routing", False),
                "routing_group_size": getattr(config, "routing_group_size", 1),
            }
            inject_block_adapters(model, adapter_config)

            # First freeze all base model parameters
            for param in model.parameters():
                param.requires_grad = False

            # Then mark block adapters as trainable
            # We need to manually set requires_grad for block adapter parameters
            for name, module in model.named_modules():
                if hasattr(module, "block_adapter"):
                    for param in module.block_adapter.parameters():
                        param.requires_grad = True
        else:
            # Normal component-level LoRA
            self._find_and_replace()
            mark_only_lora_as_trainable(self.model, self.peft_config.bias)

        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_4bit = getattr(self.model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_4bit or loaded_in_8bit:
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")
        kwargs = {
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "lora_nums": self.peft_config.lora_nums,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
            and not is_hf_device_map_available,
            "enable_blc": self.peft_config.enable_blc,
            "enable_fine_grained_routing": self.peft_config.enable_fine_grained_routing,
            "routing_group_size": self.peft_config.routing_group_size,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)
            if target_module_found:  # here
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key)
                bias = target.bias is not None

                if self.peft_config.enable_lora is None:
                    if isinstance(target, torch.nn.Linear):
                        in_features, out_features = target.in_features, target.out_features
                    elif isinstance(target, Conv1D):
                        in_features, out_features = (
                            target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                        )
                        if not kwargs.get("fan_in_fan_out", True):
                            warnings.warn(
                                "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                "Setting fan_in_fan_out to True."
                            )
                            kwargs["fan_in_fan_out"] = True
                    else:
                        raise ValueError(
                            f"Target module {target} is not supported. "
                            f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                        )

                    new_module = LoraLinear(
                        in_features,
                        out_features,
                        bias=bias,
                        **kwargs,
                    )

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key):
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False):
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = not enabled

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class LoraLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_nums: int = 2,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = True,
        enable_blc: bool = False,
        enable_fine_grained_routing: bool = False,
        routing_group_size: int = 1,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(
            self,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights,
        )

        self.lora_num = lora_nums
        self.enable_blc = enable_blc
        self.enable_fine_grained_routing = enable_fine_grained_routing
        self.routing_group_size = routing_group_size

        self.fan_in_fan_out = fan_in_fan_out

        # Actual trainable parameters
        if r > 0:
            # Fine-grained routing: output dimension-wise weights instead of scalar weights
            if self.enable_fine_grained_routing:
                # Number of groups for dimension-wise routing
                num_groups = out_features // routing_group_size
                if out_features % routing_group_size != 0:
                    raise ValueError(
                        f"out_features ({out_features}) must be divisible by routing_group_size ({routing_group_size})"
                    )
                self.lora_route = nn.Linear(in_features, self.lora_num * num_groups, bias=False)
                self.num_groups = num_groups
            else:
                # Standard scalar routing
                self.lora_route = nn.Linear(in_features, self.lora_num, bias=False)
                self.num_groups = 1

            self.lora_A = nn.Linear(in_features, r, bias=False)
            for i in range(self.lora_num):
                setattr(self, f"lora_B{i}", nn.Linear(r, out_features, bias=False))

            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)

        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            for i in range(self.lora_num):
                nn.init.zeros_(getattr(self, f"lora_B{i}").weight)

            nn.init.kaiming_uniform_(self.lora_route.weight, a=math.sqrt(5))

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        self.lora_route.train(mode)
        self.lora_A.train(mode)
        for i in range(self.lora_num):
            getattr(self, f"lora_B{i}").train(mode)

    def eval(self):
        nn.Linear.eval(self)
        self.lora_route.eval()
        self.lora_A.eval()
        for i in range(self.lora_num):
            getattr(self, f"lora_B{i}").eval()

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)[0]
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x: torch.Tensor, task_types=None):
        if self.disable_adapters:
            weight = transpose(self.weight, self.fan_in_fan_out)
            if weight.dtype != x.dtype:
                weight = weight.to(x.dtype)
            result = F.linear(x, weight, bias=self.bias)
            return result
        elif self.r > 0 and not self.merged:
            weight = transpose(self.weight, self.fan_in_fan_out)
            if weight.dtype != x.dtype:
                weight = weight.to(x.dtype)
            result = F.linear(x, weight, bias=self.bias)

            x_for_route = x.to(self.lora_route.weight.dtype) if hasattr(self.lora_route, "weight") else x
            route_logits = self.lora_route(x_for_route)

            if self.enable_fine_grained_routing:
                # Fine-grained routing: dimension-wise weights
                # route_logits: [batch, seq_len, lora_num * num_groups] -> [batch, seq_len, num_groups, lora_num]
                batch_size = route_logits.shape[0]
                if len(route_logits.shape) == 3:
                    seq_len = route_logits.shape[1]
                    route_logits = route_logits.view(batch_size, seq_len, self.num_groups, self.lora_num)
                else:
                    route_logits = route_logits.view(batch_size, self.num_groups, self.lora_num)

                route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32).to(result.dtype)

                x_for_lora = x.to(self.lora_A.weight.dtype) if hasattr(self.lora_A, "weight") else x
                lora_a_output = self.lora_A(self.lora_dropout(x_for_lora))

                for i in range(self.lora_num):
                    if len(route_weight.shape) == 4:
                        weight_i = route_weight[:, :, :, i]
                        weight_i = weight_i.repeat_interleave(self.routing_group_size, dim=-1)
                    else:
                        weight_i = route_weight[:, :, i]
                        weight_i = weight_i.repeat_interleave(self.routing_group_size, dim=-1)

                    lora_b_output = getattr(self, f"lora_B{i}")(lora_a_output)
                    result = result + (weight_i * lora_b_output * self.scaling)
            else:
                # Standard scalar routing
                route_weight = nn.functional.softmax(route_logits, dim=-1, dtype=torch.float32).to(result.dtype)

                for i in range(self.lora_num):
                    if len(route_weight.shape) == 3:
                        weight_i = route_weight[:, :, i].unsqueeze(-1)
                    else:
                        weight_i = route_weight[:, i].unsqueeze(-1)

                    x_for_lora = x.to(self.lora_A.weight.dtype) if hasattr(self.lora_A, "weight") else x
                    result = (
                        result
                        + weight_i
                        * getattr(self, f"lora_B{i}")(self.lora_A(self.lora_dropout(x_for_lora)))
                        * self.scaling
                    )

            # Calculate BLC only if explicitly enabled
            if self.enable_blc and self.training:
                # Average over all dims except the last (lora_num), works for 2D/3D/4D route_weight
                blc = self.cv_squared(route_weight.mean(dim=tuple(range(len(route_weight.shape) - 1))))
            else:
                blc = None
        else:
            weight = transpose(self.weight, self.fan_in_fan_out)
            # Ensure weight has the same dtype as input
            if weight.dtype != x.dtype:
                weight = weight.to(x.dtype)
            result = F.linear(x, weight, bias=self.bias)
            # BLC disabled - no computation
            blc = None

        # Store BLC as module attribute for collection, maintain interface compatibility
        # Only store non-None BLC values when enabled
        self.last_blc = blc
        return result
