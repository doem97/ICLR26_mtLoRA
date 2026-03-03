# here put the import lib
import importlib
import re
import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftType, _get_submodules, transpose
from .lora import LoraConfig
from .mmoelora import MMOELoraLayer, MMOELoraLinear, MMOELoraModel


def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None


@dataclass
class MMOELoraConfigS(LoraConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.MMOELora`]
    """

    task_num: int = field(default=2, metadata={"help": "The number of tasks."})
    task_embedding_dim: int = field(default=64)
    expert_num: int = field(default=4)

    def __post_init__(self):
        self.peft_type = PeftType.MMOELORAS


class MMOELoraModelS(MMOELoraModel):
    def __init__(self, model, config, adapter_name):
        super().__init__(model, config, adapter_name)

    def _find_and_replace(self, adapter_name):
        """Replace the target `Linear` module with LoRA layer (Linear+LoRA)"""
        lora_config = self.peft_config[adapter_name]
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        if loaded_in_8bit and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
            "task_num": lora_config.task_num,
            "task_embedding_dim": lora_config.task_embedding_dim,
            "expert_num": lora_config.expert_num,
        }
        key_list = [key for key, _ in self.model.named_modules()]  # all module in raw model
        for key in key_list:
            # find the corresponding modules. target module has been split into list.
            if isinstance(lora_config.target_modules, str):
                target_module_found = re.fullmatch(lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in lora_config.target_modules)
            if target_module_found:
                if not is_target_modules_in_base_model:
                    is_target_modules_in_base_model = True
                parent, target, target_name = _get_submodules(self.model, key)
                bias = target.bias is not None
                if isinstance(target, MMOELoraLayer):
                    target.update_layer(
                        adapter_name,
                        lora_config.init_r,
                        lora_config.lora_alpha,
                        lora_config.lora_dropout,
                        lora_config.init_lora_weights,
                    )
                else:
                    if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                        raise NotImplementedError
                    else:
                        if isinstance(target, torch.nn.Linear):
                            in_features, out_features = (
                                target.in_features,
                                target.out_features,
                            )
                            if kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                                    "Setting fan_in_fan_out to False."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
                        elif isinstance(target, Conv1D):
                            in_features, out_features = (
                                target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                            )
                            if not kwargs["fan_in_fan_out"]:
                                warnings.warn(
                                    "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                                    "Setting fan_in_fan_out to True."
                                )
                                kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
                        else:
                            raise ValueError(
                                f"Target module {target} is not supported. "
                                f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                            )
                        new_module = MMOELoraLinearS(
                            adapter_name,
                            in_features,
                            out_features,
                            bias=bias,
                            **kwargs,
                        )

                    self._replace_module(parent, target_name, new_module, target)
        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )


class MMOELoraLinearS(MMOELoraLinear):
    def __init__(
        self,
        adapter_name: str,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0,
        fan_in_fan_out: bool = False,
        **kwargs,
    ):
        super().__init__(
            adapter_name,
            in_features,
            out_features,
            r,
            lora_alpha,
            lora_dropout,
            fan_in_fan_out,
            **kwargs,
        )

    def unmerge(self, expert_weight):
        if self.active_adapter not in self.lora_A.keys():
            return
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        if self.r[self.active_adapter] > 0:
            for i in range(self.expert_num):
                lora_A_weights = self.lora_A[self.active_adapter].loraA[i].mlp.weight
                lora_B_weights = self.lora_B[self.active_adapter].loraB[i].mlp.weight
                self.weight.data -= (
                    transpose(
                        lora_B_weights @ lora_A_weights,
                        self.fan_in_fan_out,
                    )
                    * self.scaling[self.active_adapter]
                    * expert_weight[..., i]
                )
            self.merged = False

    def forward(self, x: torch.Tensor, **kwargs):
        # Get task_id from either 'task_id' or 'hydra_task_id' for compatibility
        task_id = kwargs.get("task_id") or kwargs.get("hydra_task_id")
        if task_id is None:
            # For inference/evaluation without explicit task_id, use task_num (the last index)
            # task_num is typically used as a "general/unknown" task in MMOELoRA
            # This is better than task_id=0 which may be biased toward a specific task type
            task_id = torch.full(
                (x.shape[0],), self.task_num, dtype=torch.long, device=x.device
            )
        previous_dtype = x.dtype

        if self.active_adapter not in self.lora_A.keys():  # No adapter, directly use linear
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        if self.disable_adapters:  # No adapter
            if self.r[self.active_adapter] > 0 and self.merged:  # merge the adapter to linear
                self.unmerge(task_id)
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        elif self.r[self.active_adapter] > 0 and not self.merged:  # general lora process
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[self.active_adapter].loraA[0].weight.dtype)

            # Get expert weights from gate network (same as MMOELoRA)
            expert_weight = self.lora_gate[self.active_adapter](self.lora_task_embedding[self.active_adapter](task_id))

            # Process through LoRA layers - A returns list, B accepts list
            lora_a_outputs = self.lora_A[self.active_adapter](self.lora_dropout[self.active_adapter](x))
            lora_b_outputs = self.lora_B[self.active_adapter](lora_a_outputs)

            # Weight each expert's output and sum
            for i in range(self.expert_num):
                # Handle different tensor dimensions for expert_weight
                # expert_weight shape: [batch, expert_num] -> weight_i shape: [batch, 1, 1]
                # Need [batch, 1, 1] to broadcast with [batch, seq, hidden]
                weight_i = expert_weight[..., i].unsqueeze(-1).unsqueeze(-1)

                result += (  # lora process
                    lora_b_outputs[i]
                    * self.scaling[self.active_adapter]
                    * weight_i
                )

            # Store last_blc for unified training pipeline interface compatibility
            # Set to None as MMOELoRAS doesn't use balance loss by default
            self.last_blc = None
        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        result = result.to(previous_dtype)

        return result
