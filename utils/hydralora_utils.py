"""
HydraLoRA utilities for Balance Loss Coefficient (BLC) collection
"""

import torch

from peft.tuners.lora import LoraLayer


def collect_balance_loss(model):
    """
    Collect Balance Loss Coefficient (BLC) from all LoRA layers and block adapters in the model.

    Args:
        model: The PEFT model containing LoRA layers or block adapters

    Returns:
        torch.Tensor or None: Average of all BLC values from enabled LoRA layers/adapters,
                             or None if no BLC values to collect
    """
    total_blc = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0

    for module in model.modules():
        # Collect from component-level LoRA layers
        if isinstance(module, LoraLayer) and hasattr(module, "last_blc"):
            if module.last_blc is not None:
                total_blc += module.last_blc
                count += 1
                # Reset BLC after collection to avoid double counting
                module.last_blc = None

        # Collect from block-level adapters
        if hasattr(module, "block_adapter") and module.block_adapter is not None:
            if hasattr(module.block_adapter, "last_blc") and module.block_adapter.last_blc is not None:
                total_blc += module.block_adapter.last_blc
                count += 1
                # Reset BLC after collection
                module.block_adapter.last_blc = None

    # Return None if no BLC to collect (all disabled)
    if count == 0:
        return None

    # Return average BLC to normalize across different model sizes
    return total_blc / count


def reset_balance_loss(model):
    """
    Reset all stored BLC values in the model.
    Useful for debugging or manual control.
    """
    for module in model.modules():
        # Reset component-level LoRA
        if isinstance(module, LoraLayer) and hasattr(module, "last_blc"):
            module.last_blc = None

        # Reset block-level adapters
        if hasattr(module, "block_adapter") and module.block_adapter is not None:
            if hasattr(module.block_adapter, "last_blc"):
                module.block_adapter.last_blc = None
