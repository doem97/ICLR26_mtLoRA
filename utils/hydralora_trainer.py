"""
Custom HuggingFace Trainer for HydraLoRA with Balance Loss support
"""

import os

import torch
from transformers import Trainer, TrainerCallback

from .hydralora_utils import collect_balance_loss


class TrainingLogCallback(TrainerCallback):
    """
    Callback to log training metrics to training.log file.
    
    HuggingFace Trainer's self.log() outputs to TensorBoard/WandB by default,
    not to the Python logging system. This callback captures training logs
    and writes them to the logging file.
    
    Note: Only logs on main process (rank 0) to avoid conflicts in DDP/DeepSpeed.
    """

    def __init__(self, log_file=None):
        self.log_file = log_file

    def _is_main_process(self, args):
        """Check if current process is the main process (rank 0)."""
        # Check various ways to determine main process
        if hasattr(args, "local_rank") and args.local_rank not in [-1, 0]:
            return False
        if hasattr(args, "process_index") and args.process_index != 0:
            return False
        # Also check environment variable for DeepSpeed
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank != 0:
            return False
        return True

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if logs is None:
            return

        # Only log on main process to avoid file write conflicts
        if not self._is_main_process(args):
            return

        # Determine log file path
        if self.log_file is None:
            if hasattr(args, "logging_dir") and args.logging_dir:
                self.log_file = os.path.join(args.logging_dir, "training.log")
            else:
                self.log_file = os.path.join(args.output_dir, "training.log")

        # Format log message
        step = state.global_step
        log_items = [f"step={step}"]
        for key, value in logs.items():
            if isinstance(value, float):
                log_items.append(f"{key}={value:.6f}")
            else:
                log_items.append(f"{key}={value}")
        log_msg = " | ".join(log_items)

        # Write to file
        with open(self.log_file, "a") as f:
            f.write(f"[TRAIN] {log_msg}\n")


class AdaLoraCallback(TrainerCallback):
    """
    Callback to handle AdaLoRA's dynamic rank allocation during training.
    It updates the total_step at the beginning and calls update_and_allocate at each step.
    """

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Handle potentially wrapped model
        if hasattr(model, "module"):
            model = model.module

        if hasattr(model, "peft_config") and hasattr(model, "trainable_adapter_name"):
            adapter_name = model.trainable_adapter_name
            if isinstance(model.peft_config, dict):
                config = model.peft_config[adapter_name]
            else:
                config = model.peft_config
            if config.peft_type == "ADALORA":
                # Set total_step from trainer state
                total_step = state.max_steps
                if hasattr(model, "rankallocator"):
                    model.rankallocator.set_total_step(total_step)
                config.total_step = total_step
                # Initialize importance scores (optional but good practice)
                model.rankallocator.reset_ipt()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Handle potentially wrapped model
        if hasattr(model, "module"):
            model = model.module

        if hasattr(model, "peft_config") and hasattr(model, "trainable_adapter_name"):
            adapter_name = model.trainable_adapter_name
            if isinstance(model.peft_config, dict):
                config = model.peft_config[adapter_name]
            else:
                config = model.peft_config
            if config.peft_type == "ADALORA":
                model.update_and_allocate(state.global_step)


class HydraLoRATrainer(Trainer):
    """
    Custom trainer that incorporates Balance Loss Coefficient (BLC) into the training process.

    This trainer automatically collects BLC from all LoRA layers after each forward pass
    and adds it to the loss with a configurable weighting factor.
    """

    def __init__(
        self,
        blc_alpha=0.01,
        enable_blc=False,
        enable_spectral_reg=False,
        spectral_reg_lambda=0.25,
        spectral_reg_frequency=1,
        spectral_reg_steps=None,
        **kwargs,
    ):
        """
        Args:
            blc_alpha (float): Weight for balance loss coefficient. Default: 0.01
            enable_blc (bool): Whether to enable BLC calculation. Default: False
            enable_spectral_reg (bool): Whether to enable spectral regularization. Default: False
            spectral_reg_lambda (float): Weight for spectral regularization loss. Default: 0.25
            spectral_reg_frequency (int): Run SVD every N epochs. Default: 1
            spectral_reg_steps (int): If set, run SVD every N steps instead of per epoch. Default: None
            **kwargs: Arguments passed to base Trainer
        """
        super().__init__(**kwargs)
        self.blc_alpha = blc_alpha
        self.enable_blc = enable_blc
        self.enable_spectral_reg = enable_spectral_reg
        self.spectral_reg_lambda = spectral_reg_lambda
        self.spectral_reg_frequency = spectral_reg_frequency
        self.spectral_reg_steps = spectral_reg_steps

        # Cache for SVD results to avoid recomputation
        self._last_svd_epoch = -1
        self._last_svd_step = -1
        self._cached_svd_data = None

        # Add TrainingLogCallback to write metrics to training.log
        self.add_callback(TrainingLogCallback)

        # Automatically add AdaLoraCallback if method is AdaLoRA
        if self.args.method == "adalora":
            self.add_callback(AdaLoraCallback)

    def _is_main_process(self):
        """Check if current process is the main process (rank 0).
        
        Works with single GPU, DDP, and DeepSpeed.
        """
        # Use Trainer's built-in method if available
        if hasattr(self.args, "local_rank") and self.args.local_rank not in [-1, 0]:
            return False
        if hasattr(self.args, "process_index") and self.args.process_index != 0:
            return False
        # Check environment variable for DeepSpeed
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return local_rank == 0

    def _should_compute_spectral_reg(self):
        """Determine if spectral regularization should be computed at this step."""
        if not self.enable_spectral_reg:
            return False

        current_step = self.state.global_step
        current_epoch = self.state.epoch if self.state.epoch is not None else 0

        # Step-based frequency takes priority
        if self.spectral_reg_steps is not None:
            if current_step - self._last_svd_step >= self.spectral_reg_steps:
                return True
        # Otherwise use epoch-based frequency
        elif int(current_epoch) - self._last_svd_epoch >= self.spectral_reg_frequency:
            return True

        return False

    def _is_using_deepspeed(self):
        """Check if we're using DeepSpeed (which has gradient reduction limitations)."""
        # Check is_deepspeed_enabled property first (most reliable)
        if hasattr(self, "is_deepspeed_enabled") and self.is_deepspeed_enabled:
            return True
        # Fallback: check if deepspeed attribute exists and is not None
        return hasattr(self, "deepspeed") and self.deepspeed is not None

    def _unwrap_model(self, model):
        """Unwrap model from DDP/PEFT wrappers to get the actual model."""
        # Handle DDP wrapper
        if hasattr(model, "module"):
            model = model.module
        # Handle PEFT wrapper (PeftModel -> base_model -> model)
        if hasattr(model, "base_model"):
            model = model.base_model
        if hasattr(model, "model"):
            model = model.model
        return model

    def compute_spectral_regularization(self, model):
        """
        Compute spectral-aware orthogonal regularization on B matrices.

        This encourages each expert's B matrix to have well-conditioned columns,
        weighted by spectral importance (low-SV components get stronger regularization).

        L = Σ_i ||W @ (B_i^T @ B_i - I·scale)||_F²

        NOTE: DeepSpeed ZeRO-2 doesn't support multiple gradient paths to the same
        parameter. When DeepSpeed is detected, this returns a monitoring-only value.
        Use DDP for experiments that need spectral regularization to affect training.

        Returns:
            torch.Tensor: Spectral regularization loss
        """
        # Unwrap model from DDP/PEFT wrappers
        unwrapped_model = self._unwrap_model(model)

        # Collect all B matrices from LoRA layers and block adapters
        b_matrices = []
        for name, module in unwrapped_model.named_modules():
            # Collect from component-level LoRA layers
            if hasattr(module, "lora_num") and hasattr(module, "lora_A"):
                for i in range(module.lora_num):
                    b_layer = getattr(module, f"lora_B{i}", None)
                    if b_layer is not None and hasattr(b_layer, "weight"):
                        b_matrices.append(b_layer.weight)

            # Collect from block-level adapters
            if hasattr(module, "block_adapter") and module.block_adapter is not None:
                adapter = module.block_adapter
                if hasattr(adapter, "lora_B") and isinstance(adapter.lora_B, torch.nn.ModuleList):
                    for b_layer in adapter.lora_B:
                        if hasattr(b_layer, "weight"):
                            b_matrices.append(b_layer.weight)

        if len(b_matrices) == 0:
            # Log warning only once per training run (main process only)
            if not hasattr(self, "_warned_no_b_matrices") and self._is_main_process():
                print(f"[SpectralReg] WARNING: No B matrices found! Check model structure.")
                self._warned_no_b_matrices = True
            return torch.tensor(0.0, device=next(model.parameters()).device)

        # Log success info only once (main process only)
        if not hasattr(self, "_logged_b_matrices_info") and self._is_main_process():
            print(f"[SpectralReg] Found {len(b_matrices)} B matrices for regularization")
            self._logged_b_matrices_info = True

        # Sample subset if too many matrices
        if len(b_matrices) > 64:
            import random

            random.seed(self.state.global_step)
            b_matrices = random.sample(b_matrices, 64)

        # Check if using DeepSpeed - if so, compute monitoring-only (no gradients)
        use_deepspeed = self._is_using_deepspeed()

        if use_deepspeed:
            # # DeepSpeed mode: monitoring only (detached computation)
            # total_loss = 0.0
            # with torch.no_grad():
            #     for b_matrix in b_matrices:
            #         rank = b_matrix.shape[1]
            #         b_detached = b_matrix.detach().float()
            #         gram = torch.mm(b_detached.t(), b_detached)

            #         try:
            #             S = torch.linalg.svdvals(b_detached)
            #             sigma_mean = S.mean() + 1e-8
            #             weights = torch.exp(-S / sigma_mean)
            #         except Exception:
            #             weights = torch.ones(rank, device=b_matrix.device)

            #         scale = torch.linalg.norm(b_detached).item() / (rank**0.5) + 1e-8
            #         identity = torch.eye(rank, device=b_matrix.device) * (scale**2)
            #         diff = gram - identity
            #         weighted_diff = diff * weights.unsqueeze(0) * weights.unsqueeze(1)
            #         total_loss += (weighted_diff**2).sum().item()

            # total_loss = total_loss / len(b_matrices)
            # result = torch.tensor(total_loss, device=next(model.parameters()).device)
            raise ValueError(
                "Spectral Regularization is not compatible with DeepSpeed (ZeRO-2/3) as it requires multiple gradient paths. "
                "Please disable DeepSpeed or disable spectral regularization."
            )
        else:
            # DDP mode: full gradient computation
            total_loss = torch.tensor(0.0, device=b_matrices[0].device, dtype=b_matrices[0].dtype)

            for b_matrix in b_matrices:
                rank = b_matrix.shape[1]

                # Compute Gram matrix (with gradients)
                gram = torch.mm(b_matrix.t(), b_matrix)  # [rank, rank]

                # Compute spectral weights (detached - just for weighting)
                with torch.no_grad():
                    try:
                        S = torch.linalg.svdvals(b_matrix.detach().float())
                        sigma_mean = S.mean() + 1e-8
                        weights = torch.exp(-S / sigma_mean).to(b_matrix.dtype)
                    except Exception:
                        weights = torch.ones(rank, device=b_matrix.device, dtype=b_matrix.dtype)

                    scale = torch.linalg.norm(b_matrix.detach()).item() / (rank**0.5) + 1e-8

                # Identity target (scaled)
                identity = torch.eye(rank, device=b_matrix.device, dtype=b_matrix.dtype) * (scale**2)

                # Weighted deviation from identity
                diff = gram - identity
                weighted_diff = diff * weights.unsqueeze(0) * weights.unsqueeze(1)

                # Frobenius norm squared
                ortho_loss = (weighted_diff**2).sum()
                total_loss = total_loss + ortho_loss

            result = total_loss / len(b_matrices)

        # Update cache tracking
        self._last_svd_step = self.state.global_step
        if self.state.epoch is not None:
            self._last_svd_epoch = int(self.state.epoch)

        return result

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Compute loss with Balance Loss Coefficient included.

        Args:
            model: The model
            inputs: The inputs dict
            return_outputs: Whether to return model outputs

        Returns:
            loss (with BLC) or (loss, outputs) if return_outputs=True
        """
        # Systemic Fix: Filter out task_types for methods that don't support routing (AdaLoRA, Standard LoRA)
        if self.args.method not in ["hydralora", "mmoelora", "mmoeloras"] and "task_types" in inputs:
            inputs.pop("task_types")

        labels = inputs.get("labels")
        # Forward pass
        outputs = model(**inputs)

        # Compute primary loss
        if labels is not None:
            loss = outputs.loss
        else:
            # If no labels, compute loss from logits and labels in inputs
            if self.label_smoother is not None and "labels" in inputs:
                loss = self.label_smoother(outputs, inputs["labels"])
            else:
                logits = outputs.get("logits")
                if logits is not None:
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)), inputs["labels"].view(-1)
                    )
                else:
                    raise ValueError("No loss or logits found in model outputs")

        # Collect Balance Loss Coefficient only if enabled
        total_loss = loss
        balance_loss_value = None
        spectral_loss_value = None

        if self.enable_blc and self.blc_alpha > 0:
            balance_loss = collect_balance_loss(model)
            if balance_loss is not None:
                total_loss = total_loss + self.blc_alpha * balance_loss
                balance_loss_value = balance_loss.item()

        # Compute spectral regularization if enabled and frequency condition met
        # NOTE: With DeepSpeed ZeRO-2, spectral loss is monitoring-only (no gradient contribution).
        # With DDP, spectral loss contributes to gradients normally.
        if self._should_compute_spectral_reg():
            spectral_loss = self.compute_spectral_regularization(model)
            if spectral_loss is not None:
                spectral_loss_value = spectral_loss.item()
                # Only add to total_loss if it has gradients (DDP mode)
                if spectral_loss.requires_grad:
                    total_loss = total_loss + self.spectral_reg_lambda * spectral_loss
                    # Log when spectral reg is triggered (main process only)
                    if self._is_main_process():
                        print(
                            f"[SpectralReg] Step {self.state.global_step}: "
                            f"spectral_loss={spectral_loss_value:.6f}, "
                            f"weighted={self.spectral_reg_lambda * spectral_loss_value:.6f}"
                        )
                else:
                    # Log warning if no gradients (shouldn't happen in DDP mode)
                    if not hasattr(self, "_warned_no_grad") and self._is_main_process():
                        print(
                            f"[SpectralReg] WARNING: spectral_loss has no gradients at step {self.state.global_step}. "
                            "This may indicate B matrices were not found or DeepSpeed is enabled."
                        )
                        self._warned_no_grad = True

        # Log losses periodically for monitoring
        if self.state.global_step % 50 == 0:
            log_dict = {"task_loss": loss.item(), "total_loss": total_loss.item()}
            if balance_loss_value is not None:
                log_dict["balance_loss"] = balance_loss_value
            if spectral_loss_value is not None:
                log_dict["spectral_loss"] = spectral_loss_value
            self.log(log_dict)

        return (total_loss, outputs) if return_outputs else total_loss
