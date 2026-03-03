#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import torch
import transformers
from peft import (
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
)
from peft.tuners.lora import LoraLayer
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    LlamaForCausalLM,
    LlamaTokenizer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    get_last_checkpoint,
)
from transformers.utils import send_example_telemetry
from utils import DataCollatorForSupervisedDataset, build_instruction_dataset
from utils.hydralora_trainer import HydraLoRATrainer
from utils.method_configs import get_method_config


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "sft_lora_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "sft_lora_model")
        model_to_save = kwargs.get("model")
        if model_to_save:
            model_to_save.save_pretrained(peft_model_path)
        else:
            logger.warning(
                "Model not found in SavePeftModelCallback kwargs during on_save."
            )

        tokenizer_to_save = kwargs.get("tokenizer")
        if tokenizer_to_save:
            tokenizer_to_save.save_pretrained(peft_model_path)
        else:
            logger.warning(
                "Tokenizer not found in SavePeftModelCallback kwargs during on_save."
            )

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        model_to_save = kwargs.get("model")
        tokenizer_to_save = kwargs.get("tokenizer")

        if model_to_save and tokenizer_to_save:
            peft_model_path = os.path.join(args.output_dir, "sft_lora_model")
            model_to_save.save_pretrained(peft_model_path)
            tokenizer_to_save.save_pretrained(peft_model_path)
        else:
            logger.warning(
                "Model or tokenizer not available inSavePeftModelCallback kwargs during on_train_end."
            )


def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(
        model, "is_loaded_in_4bit", False
    )

    for _name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8/INT4 parameters to fp32
    for param in model.parameters():
        if (
            (param.dtype == torch.float16) or (param.dtype == torch.bfloat16)
        ) and loaded_in_kbit:
            param.data = param.data.to(torch.float32)

    for name, module in model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "The tokenizer for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )

    config_overrides: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: str | None = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: str | None = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: str | None = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: str | None = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_dir: str | None = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )

    train_file: str | None = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."},
    )
    validation_file: str | None = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )

    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: float | None = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )
    data_cache_dir: str | None = field(
        default=None, metadata={"help": "The datasets processed stored"}
    )

    max_seq_length: int | None = field(default=1024)


@dataclass
class MyTrainingArguments(TrainingArguments):
    method: str = field(
        default="lora",
        metadata={
            "help": "LoRA method to use. Choices: lora, mtlora, hydralora, mmoelora, mmoeloras, adalora"
        },
    )
    trainable: str | None = field(default="q_proj,v_proj")
    lora_rank: int | None = field(default=8)
    lora_dropout: float | None = field(default=0.1)
    lora_alpha: float | None = field(default=32.0)
    modules_to_save: str | None = field(default=None)
    peft_path: str | None = field(default=None)
    flash_attn: bool | None = field(default=False)
    double_quant: bool | None = field(default=True)
    quant_type: str | None = field(default="nf4")
    load_in_kbits: int | None = field(default=16)

    # HydraLoRA specific
    lora_nums: int | None = field(default=2)
    enable_blc: bool = field(
        default=False,
        metadata={
            "help": "Enable Balance Loss Coefficient calculation. Default: False to avoid OOM"
        },
    )
    blc_alpha: float | None = field(
        default=0.01, metadata={"help": "Weight for balance loss coefficient"}
    )
    logging_dir: str | None = field(
        default=None,
        metadata={
            "help": "Optional separate directory for log files. If not set, logs are saved in output_dir."
        },
    )

    # MMOE specific
    task_num: int | None = field(
        default=2, metadata={"help": "Number of tasks for MMOE methods"}
    )
    task_embedding_dim: int | None = field(
        default=64, metadata={"help": "Task embedding dimension for MMOE"}
    )
    expert_num: int | None = field(
        default=4, metadata={"help": "Number of experts for MMOE"}
    )

    # AdaLoRA specific
    target_r: int | None = field(
        default=8, metadata={"help": "Target rank for AdaLoRA"}
    )
    init_r: int | None = field(
        default=12, metadata={"help": "Initial rank for AdaLoRA"}
    )
    orth_reg_weight: float | None = field(
        default=0.5,
        metadata={
            "help": "Orthogonal regularization weight for AdaLoRA (default 0.5, standard value)"
        },
    )

    # mtLoRA specific (fine-grained routing + spectral regularization)
    enable_fine_grained_routing: bool = field(
        default=False,
        metadata={"help": "Enable fine-grained routing with dimension-wise weights"},
    )
    routing_group_size: int | None = field(
        default=1,
        metadata={
            "help": "Group size for fine-grained routing (1=full dimension-wise)"
        },
    )
    enable_spectral_reg: bool = field(
        default=False,
        metadata={"help": "Enable spectral-aware orthogonal regularization"},
    )
    spectral_reg_lambda: float | None = field(
        default=0.25, metadata={"help": "Weight for spectral regularization loss"}
    )
    spectral_reg_frequency: int | None = field(
        default=1,
        metadata={"help": "Run SVD every N epochs for spectral regularization"},
    )
    spectral_reg_steps: int | None = field(
        default=None,
        metadata={"help": "If set, also run SVD every N steps (overrides epoch-based)"},
    )

    # Block-level adaptation (mtLoRA third component)
    enable_block_adapter: bool = field(
        default=False,
        metadata={
            "help": "Enable block-level adaptation instead of component-level LoRA"
        },
    )
    block_adapter_type: str = field(
        default="attention",
        metadata={
            "help": "Where to apply block adapter: 'attention', 'ffn', or 'both'"
        },
    )
    block_adapter_style: str = field(
        default="lowrank",
        metadata={"help": "Adapter style: 'lowrank', 'bottleneck', or 'adaptformer'"},
    )
    block_adapter_rank: int | None = field(
        default=None,
        metadata={"help": "Adapter rank (defaults to lora_rank if not set)"},
    )
    adaptformer_init_scale: float = field(
        default=0.0,
        metadata={
            "help": "Initial value for AdaptFormer per-expert learnable scales (default 0.0 for zero-init)"
        },
    )


logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, MyTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Apply HydraLoRA patches to transformers library ONLY if needed
    # Optimization: AdaLoRA/Standard LoRA don't need routing patches,
    # skipping them avoids potential memory overhead/leaks.
    if training_args.method in ["mtlora", "hydralora", "mmoelora", "mmoeloras"]:
        from peft.utils.transformers_patch import patch_llama_for_hydralora

        patch_llama_for_hydralora()

    if training_args.flash_attn:
        raise NotImplementedError("Flash attention is not implemented")
        # from flash_attn_patch import replace_llama_attn_with_flash_attn
        # replace_llama_attn_with_flash_attn()

    send_example_telemetry("run_clm", model_args, data_args)

    # Setup simple logging: console + single shared file
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Handle logging directory logic
    if training_args.logging_dir is not None:
        os.makedirs(training_args.logging_dir, exist_ok=True)
        log_file = os.path.join(training_args.logging_dir, "training.log")
    else:
        log_file = os.path.join(training_args.output_dir, "training.log")

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="a"),
    ]

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=handlers,
        force=True,
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    # transformers.tokenization_utils.logging.set_verbosity_warning()

    # Log on each process the small summary:
    # Use world_size > 1 as proper distributed training indicator instead of local_rank != -1
    # because new HuggingFace Transformers always sets local_rank=0 for single process
    is_distributed = training_args.world_size > 1
    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"world_size: {training_args.world_size}, "
        f"distributed training: {is_distributed}, "
        f"16-bits training: {training_args.fp16 or training_args.bf16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        print("last_checkpoint", last_checkpoint)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<unk>",
    }

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.tokenizer_name_or_path:
        tokenizer = LlamaTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path, **tokenizer_kwargs
        )
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        assert tokenizer.pad_token == "<unk>"
        assert tokenizer.pad_token_id == 0
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    # if (len(tokenizer)) != 55296:
    #     raise ValueError(f"The vocab size of the tokenizer should be 55296, but found {len(tokenizer)}.\n"
    #                      "Please use Chinese-LLaMA-2 tokenizer.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    eval_dataset = None
    train_dataset = None

    if training_args.do_train:
        with training_args.main_process_first(desc="loading and tokenization"):
            path = Path(data_args.dataset_dir)
            files = [os.path.join(path, file.name) for file in path.glob("*.json")]
            logger.info(f"Training files: {' '.join(files)}")
            logger.info(f"[CONFIG] max_seq_length={data_args.max_seq_length}")
            train_dataset = build_instruction_dataset(
                data_path=files,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                data_cache_dir=None,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
            )
        logger.info(f"Num train_samples  {len(train_dataset)}")
        # TAG: print first training sample for debug
        # logger.info(f"Training example input: {tokenizer.decode(train_dataset[0]['input_ids'])}")
        # logger.info(f"Training example: {train_dataset[0]}")
    if training_args.do_eval:
        with training_args.main_process_first(desc="loading and tokenization"):
            files = [data_args.validation_file]
            logger.info(f"Evaluation files: {' '.join(files)}")
            logger.info(f"[CONFIG] max_seq_length={data_args.max_seq_length}")
            eval_dataset = build_instruction_dataset(
                data_path=files,
                tokenizer=tokenizer,
                max_seq_length=data_args.max_seq_length,
                data_cache_dir=None,
                preprocessing_num_workers=data_args.preprocessing_num_workers,
            )
        logger.info(f"Num eval_samples  {len(eval_dataset)}")
        # TAG: print first eval sample for debug
        # logger.info(f"Evaluation example input: {tokenizer.decode(eval_dataset[0]['input_ids'])}")
        # logger.info(f"Evaluation example: {eval_dataset[0]}")

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    if training_args.load_in_kbits in [4, 8]:
        load_in_4bit = training_args.load_in_kbits == 4
        load_in_8bit = training_args.load_in_kbits == 8
        if training_args.modules_to_save is not None:
            load_in_8bit_skip_modules = training_args.modules_to_save.split(",")
        else:
            load_in_8bit_skip_modules = None
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=training_args.load_in_kbits == 4,
            load_in_8bit=training_args.load_in_kbits == 8,
            llm_int8_threshold=6.0,
            load_in_8bit_skip_modules=load_in_8bit_skip_modules,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
        )
    else:
        load_in_4bit = False
        load_in_8bit = False
        quantization_config = None
    if quantization_config is not None:
        logger.info(f"quantization_config:{quantization_config.to_dict()}")

    # TAG: ZeRO-3 will handle device_map, do not set it
    # device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    logger.info(
        f"[CHECKPOINT] Starting model load from {model_args.model_name_or_path}"
    )
    logger.info(
        f"[CONFIG] DeepSpeed: {training_args.deepspeed is not None}, "
        f"FP16: {training_args.fp16}, BF16: {training_args.bf16}"
    )
    logger.info(
        f"[CONFIG] World size: {training_args.world_size}, Local rank: {training_args.local_rank}"
    )

    if training_args.deepspeed:
        logger.info(f"[DEEPSPEED] Config file: {training_args.deepspeed}")
        try:
            import deepspeed

            logger.info(f"[DEEPSPEED] Version: {deepspeed.__version__}")
        except ImportError:
            logger.warning("[DEEPSPEED] DeepSpeed not available")

    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch_dtype,
        # low_cpu_mem_usage=True,
        # device_map=device_map,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        quantization_config=quantization_config,
    )
    logger.info(
        f"[CHECKPOINT] Model loaded successfully, parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    model.enable_input_require_grads()
    if training_args.load_in_kbits in [4, 8]:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )
    model.config.use_cache = False

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    logger.info(f"Model vocab size: {model_vocab_size}")
    logger.info(f"len(tokenizer):{len(tokenizer)}")
    if model_vocab_size != len(tokenizer):
        new_vocab_size = len(tokenizer)
        logger.info(
            f"Resize model vocab size to {new_vocab_size}, padding to multiple of 64."
        )
        model.resize_token_embeddings(new_vocab_size, pad_to_multiple_of=64)
        config.vocab_size = model.get_input_embeddings().weight.shape[0]
        logger.info(f"New model vocab size after padding: {config.vocab_size}")

    if (
        training_args.peft_path is not None
    ):  # --------------------------> train from the trained lora model
        logger.info("Peft from pre-trained model")

        model = PeftModel.from_pretrained(
            model,
            training_args.peft_path,
            # device_map=device_map
        )
    else:  # --------------------------> train from the scratch
        logger.info("Init new peft model")
        target_modules = training_args.trainable.split(",")  # lora paras
        modules_to_save = (
            training_args.modules_to_save
        )  # not lora paras, but is trainable, i.e., not freeze
        if modules_to_save is not None:
            modules_to_save = modules_to_save.split(",")
        # Create method-specific configuration
        logger.info(f"[CHECKPOINT] Initializing {training_args.method.upper()} method")

        # Build configuration based on selected method
        method_kwargs = {
            "r": training_args.lora_rank,
            "lora_alpha": training_args.lora_alpha,
            "lora_dropout": training_args.lora_dropout,
            "target_modules": target_modules,
            "modules_to_save": modules_to_save,
        }

        # Add method-specific parameters
        if training_args.method in ["mtlora", "hydralora"]:
            method_kwargs.update(
                {
                    "lora_nums": training_args.lora_nums,
                    "enable_blc": training_args.enable_blc,
                    "blc_alpha": training_args.blc_alpha,
                    "enable_fine_grained_routing": training_args.enable_fine_grained_routing,
                    "routing_group_size": training_args.routing_group_size,
                    "enable_spectral_reg": training_args.enable_spectral_reg,
                    "spectral_reg_lambda": training_args.spectral_reg_lambda,
                    "spectral_reg_frequency": training_args.spectral_reg_frequency,
                    "spectral_reg_steps": training_args.spectral_reg_steps,
                    "enable_block_adapter": training_args.enable_block_adapter,
                    "block_adapter_type": training_args.block_adapter_type,
                    "block_adapter_style": training_args.block_adapter_style,
                    "block_adapter_rank": training_args.block_adapter_rank
                    or training_args.lora_rank,
                    "adaptformer_init_scale": training_args.adaptformer_init_scale,
                }
            )
            logger.info(
                f"HydraLoRA config: rank={training_args.lora_rank}, "
                f"experts={training_args.lora_nums}, BLC={training_args.enable_blc}, "
                f"fine-grained routing={training_args.enable_fine_grained_routing}, "
                f"spectral reg={training_args.enable_spectral_reg}, "
                f"block adapter={training_args.enable_block_adapter} (type={training_args.block_adapter_type}, style={training_args.block_adapter_style})"
            )
            if training_args.block_adapter_style == "adaptformer":
                logger.info(
                    f"AdaptFormer init_scale: {training_args.adaptformer_init_scale}"
                )
        elif training_args.method in ["mmoelora", "mmoeloras"]:
            method_kwargs.update(
                {
                    "task_num": training_args.task_num,
                    "task_embedding_dim": training_args.task_embedding_dim,
                    "expert_num": training_args.expert_num,
                }
            )
            logger.info(
                f"MMOE config: tasks={training_args.task_num}, experts={training_args.expert_num}"
            )
        elif training_args.method == "adalora":
            method_kwargs.update(
                {
                    "target_r": training_args.target_r,
                    "init_r": training_args.init_r,
                    "orth_reg_weight": training_args.orth_reg_weight,
                }
            )
            logger.info(
                f"AdaLoRA config: init_r={training_args.init_r}, target_r={training_args.target_r}, "
                f"orth_reg_weight={training_args.orth_reg_weight}"
            )
        else:  # Standard LoRA
            logger.info(f"Standard LoRA config: rank={training_args.lora_rank}")

        # Get method configuration and convert to PEFT config
        method_config = get_method_config(training_args.method, **method_kwargs)
        peft_config = method_config.to_peft_config()

        logger.info(f"Target modules: {target_modules}")

        model = get_peft_model(model, peft_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[CHECKPOINT] PEFT model created: {trainable_params:,} trainable / "
        f"{all_params:,} total ({100 * trainable_params / all_params:.2f}%)"
    )

    if training_args.gradient_checkpointing and (
        not model.modules_to_save or "embed_tokens" not in model.modules_to_save
    ):
        # enable requires_grad to avoid error in backward pass with gradient_checkpoint
        if hasattr(model.base_model, "enable_input_require_grads"):
            model.base_model.enable_input_require_grads()
        elif hasattr(model.base_model, "get_input_embeddings"):

            def make_inputs_require_grad(_module, _input, _output):
                _output.requires_grad_(True)
                # print(
                #     f"[DEBUG] Hook make_inputs_require_grad called."
                #     f"Output tensor: {_output.shape},"
                #     f"requires_grad: {_output.requires_grad}"
                # )

            model.base_model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad
            )
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if training_args.bf16:
                module = module.to(torch.bfloat16)
            if training_args.fp16:
                module = module.to(torch.float16)
        if "norm" in name:
            module = module.to(torch.float16)
        if ("lm_head" in name or "embed_tokens" in name) and hasattr(module, "weight"):
            if training_args.bf16 and module.weight.dtype == torch.float32:
                module = module.to(torch.bfloat16)
            if training_args.fp16 and module.weight.dtype == torch.float32:
                module = module.to(torch.float16)
    model.print_trainable_parameters()
    logger.info(f"model.modules_to_save: {model.modules_to_save}")
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
    ).__get__(model, type(model))

    # TAG: only print those who to train
    # for name, parameters in model.named_parameters():
    #     if parameters.requires_grad:
    #         logger.info(f"{name}, :, {parameters.size()},{parameters.requires_grad}")

    training_args.remove_unused_columns = False

    logger.info(
        f"[CHECKPOINT] Creating HydraLoRATrainer with batch_size={training_args.per_device_train_batch_size}, "
        f"grad_accum={training_args.gradient_accumulation_steps}, "
        f"enable_blc={training_args.enable_blc}, blc_alpha={training_args.blc_alpha}, "
        f"enable_spectral_reg={training_args.enable_spectral_reg}, "
        f"spectral_reg_lambda={training_args.spectral_reg_lambda}, "
        f"spectral_reg_steps={training_args.spectral_reg_steps}"
    )
    # Initialize our HydraLoRA Trainer with Balance Loss support
    trainer = HydraLoRATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        enable_blc=training_args.enable_blc,
        blc_alpha=training_args.blc_alpha,
        enable_spectral_reg=training_args.enable_spectral_reg,
        spectral_reg_lambda=training_args.spectral_reg_lambda,
        spectral_reg_frequency=training_args.spectral_reg_frequency,
        spectral_reg_steps=training_args.spectral_reg_steps,
    )

    trainer.add_callback(SavePeftModelCallback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        logger.info(
            f"[CHECKPOINT] Starting training, max_steps={training_args.max_steps}, checkpoint={checkpoint}"
        )
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics

        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # If logging_dir is set, also save metrics and state there for Git tracking
        if training_args.logging_dir:
            with open(
                os.path.join(training_args.logging_dir, "train_results.json"), "w"
            ) as f:
                import json

                json.dump(metrics, f, indent=2)
            # Also copy the trainer_state.json if possible or just verify it's logged
            # Note: save_state() saves to output_dir/trainer_state.json by default.
            # We manually copy it to logging_dir/trainer_state.json
            import shutil

            state_path = os.path.join(training_args.output_dir, "trainer_state.json")
            if os.path.exists(state_path):
                shutil.copy2(
                    state_path,
                    os.path.join(training_args.logging_dir, "trainer_state.json"),
                )


if __name__ == "__main__":
    main()
