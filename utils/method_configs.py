"""
Unified configuration system for multi-LoRA method comparison.

This module provides a clean interface for configuring different LoRA variants
with their method-specific hyperparameters exposed clearly.
"""

from dataclasses import dataclass, field

from peft import LoraConfig, TaskType
from peft.tuners.adalora import AdaLoraConfig
from peft.tuners.mmoelora import MMOELoraConfig
from peft.tuners.mmoeloraS import MMOELoraConfigS
from peft.utils import PeftType


@dataclass
class BaseMethodConfig:
    """Base configuration shared by all methods."""

    method_name: str
    r: int = 8  # LoRA rank
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] | str | None = None
    modules_to_save: list[str] | None = None
    bias: str = "none"

    def to_peft_config(self):
        """Convert to appropriate PEFT configuration."""
        raise NotImplementedError


@dataclass
class StandardLoRAConfig(BaseMethodConfig):
    """Configuration for standard LoRA."""

    method_name: str = field(default="lora", init=False)

    def to_peft_config(self):
        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            modules_to_save=self.modules_to_save,
            bias=self.bias,
            lora_nums=1,  # Standard LoRA has single expert
            enable_blc=False,
            task_type=TaskType.CAUSAL_LM,
        )


@dataclass
class HydraLoRAConfig(BaseMethodConfig):
    """Configuration for HydraLoRA with multiple B matrices and routing."""

    method_name: str = field(default="hydralora", init=False)
    lora_nums: int = 4  # Number of B matrix experts
    enable_blc: bool = False  # Balance Loss Coefficient
    blc_alpha: float = 0.01  # BLC weight

    # mtLoRA extensions
    enable_fine_grained_routing: bool = False
    routing_group_size: int = 1
    enable_spectral_reg: bool = False
    spectral_reg_lambda: float = 0.25
    spectral_reg_frequency: int = 1
    spectral_reg_steps: int | None = None
    enable_block_adapter: bool = False
    block_adapter_type: str = "attention"
    block_adapter_style: str = "lowrank"
    block_adapter_rank: int | None = None
    adaptformer_init_scale: float = 0.0

    def to_peft_config(self):
        config = LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            modules_to_save=self.modules_to_save,
            bias=self.bias,
            lora_nums=self.lora_nums,
            enable_blc=self.enable_blc,
            enable_fine_grained_routing=self.enable_fine_grained_routing,
            routing_group_size=self.routing_group_size,
            enable_spectral_reg=self.enable_spectral_reg,
            spectral_reg_lambda=self.spectral_reg_lambda,
            spectral_reg_frequency=self.spectral_reg_frequency,
            spectral_reg_steps=self.spectral_reg_steps,
            enable_block_adapter=self.enable_block_adapter,
            block_adapter_type=self.block_adapter_type,
            block_adapter_style=self.block_adapter_style,
            block_adapter_rank=self.block_adapter_rank or self.r,
            adaptformer_init_scale=self.adaptformer_init_scale,
            task_type=TaskType.CAUSAL_LM,
        )
        # Mark as HydraLoRA explicitly
        config.peft_type = PeftType.HYDRALORA
        return config


@dataclass
class MMOELoRAConfig(BaseMethodConfig):
    """Configuration for Multi-gate Mixture of Experts LoRA."""

    method_name: str = field(default="mmoelora", init=False)
    task_num: int = 2  # Number of tasks
    task_embedding_dim: int = 64
    expert_num: int = 4  # Number of experts

    def to_peft_config(self):
        return MMOELoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            modules_to_save=self.modules_to_save,
            bias=self.bias,
            task_num=self.task_num,
            task_embedding_dim=self.task_embedding_dim,
            expert_num=self.expert_num,
            task_type=TaskType.CAUSAL_LM,
        )


@dataclass
class MMOELoRASConfig(BaseMethodConfig):
    """Configuration for simplified MMOE LoRA variant."""

    method_name: str = field(default="mmoeloras", init=False)
    task_num: int = 2
    task_embedding_dim: int = 64
    expert_num: int = 4

    def to_peft_config(self):
        return MMOELoraConfigS(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            modules_to_save=self.modules_to_save,
            bias=self.bias,
            task_num=self.task_num,
            task_embedding_dim=self.task_embedding_dim,
            expert_num=self.expert_num,
            task_type=TaskType.CAUSAL_LM,
        )


@dataclass
class AdaLoRAMethodConfig(BaseMethodConfig):
    """Configuration for Adaptive LoRA."""

    method_name: str = field(default="adalora", init=False)
    target_r: int = 8  # Target average rank
    init_r: int = 12  # Initial rank
    tinit: int = 0  # Warmup steps
    tfinal: int = 0  # Final steps
    deltaT: int = 1  # Update interval
    beta1: float = 0.85
    beta2: float = 0.85
    orth_reg_weight: float = 0.5

    def to_peft_config(self):
        return AdaLoraConfig(
            r=self.init_r,
            init_r=self.init_r,
            target_r=self.target_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            modules_to_save=self.modules_to_save,
            bias=self.bias,
            tinit=self.tinit,
            tfinal=self.tfinal,
            deltaT=self.deltaT,
            beta1=self.beta1,
            beta2=self.beta2,
            orth_reg_weight=self.orth_reg_weight,
            task_type=TaskType.CAUSAL_LM,
        )


METHOD_CONFIGS = {
    "lora": StandardLoRAConfig,
    "hydralora": HydraLoRAConfig,
    "mtlora": HydraLoRAConfig,  # mtlora is an alias for hydralora
    "mmoelora": MMOELoRAConfig,
    "mmoeloras": MMOELoRASConfig,
    "adalora": AdaLoRAMethodConfig,
}


def get_method_config(method_name: str, **kwargs) -> BaseMethodConfig:
    """
    Factory function to create method configuration.

    Args:
        method_name: Name of the method (lora, hydralora, mmoelora, etc.)
        **kwargs: Method-specific configuration parameters

    Returns:
        Configured method instance
    """
    if method_name not in METHOD_CONFIGS:
        raise ValueError(
            f"Unknown method: {method_name}. Available: {list(METHOD_CONFIGS.keys())}"
        )

    config_class = METHOD_CONFIGS[method_name]
    return config_class(**kwargs)


def add_method_arguments(parser, method_name: str):
    """
    Add method-specific arguments to argument parser.

    Args:
        parser: ArgumentParser instance
        method_name: Name of the method
    """
    # Common LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA scaling factor"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="LoRA dropout rate"
    )

    # Method-specific arguments
    if method_name == "hydralora":
        parser.add_argument(
            "--lora_nums",
            type=int,
            default=4,
            help="Number of B matrix experts for HydraLoRA",
        )
        parser.add_argument(
            "--enable_blc", action="store_true", help="Enable Balance Loss Coefficient"
        )
        parser.add_argument(
            "--blc_alpha", type=float, default=0.01, help="BLC weight factor"
        )

    elif method_name in ["mmoelora", "mmoeloras"]:
        parser.add_argument(
            "--task_num", type=int, default=2, help="Number of tasks for MMOE"
        )
        parser.add_argument(
            "--task_embedding_dim",
            type=int,
            default=64,
            help="Task embedding dimension",
        )
        parser.add_argument(
            "--expert_num", type=int, default=4, help="Number of experts in MMOE"
        )

    elif method_name == "adalora":
        parser.add_argument(
            "--target_r", type=int, default=8, help="Target average rank for AdaLoRA"
        )
        parser.add_argument(
            "--init_r", type=int, default=12, help="Initial rank for AdaLoRA"
        )
        parser.add_argument(
            "--tinit", type=int, default=0, help="Warmup steps for AdaLoRA"
        )
        parser.add_argument(
            "--tfinal", type=int, default=0, help="Final steps for AdaLoRA"
        )
        parser.add_argument(
            "--deltaT", type=int, default=1, help="Update interval for AdaLoRA"
        )
        parser.add_argument(
            "--beta1", type=float, default=0.85, help="Beta1 for AdaLoRA"
        )
        parser.add_argument(
            "--beta2", type=float, default=0.85, help="Beta2 for AdaLoRA"
        )
        parser.add_argument(
            "--orth_reg_weight",
            type=float,
            default=0.5,
            help="Orthogonal regularization weight",
        )
