# mtLoRA: Scalable Multi-Task Low-Rank Adaptation

Official implementation for **"Scalable Multi-Task Low-Rank Adaptation"**.

mtLoRA addresses the fundamental scalability challenge in multi-task LoRA adaptation through three key designs:
1. **Spectral-Aware Regularization** - Selectively orthogonalizes low-SV components while preserving high-SV shared knowledge
2. **Fine-Grained Routing** - Dimension-specific routing weights instead of scalar weights per LoRA
3. **Block-Level Adaptation** - Applies LoRA at block level to mitigate gradient conflict amplification

## Quick Start

> [!TIP]
> We provide **Claude Code agent** for easier reproduction. See [Reproduce with Claude Agent](#reproduce-with-claude-agent) section.

```bash
# 1. Setup environment
conda env create -f environment.yml
conda activate mtlora
pip install -e ./peft

# 2. Setup data (see Data Preparation below)

# 3. Run experiments
bash tables/0_main_ablation.sh
```

## Reproduce with Claude Agent

For guided reproduction, use the Claude Code agent in Cursor or Claude Code:

1. Open this project in Cursor with Claude enabled (or use Claude Code CLI)
2. Claude reads `CLAUDE.md` and assists with:
   - Environment setup based on your GPU/CUDA version
   - Dataset download and configuration
   - Running specific table reproductions

Just ask: *"Help me reproduce Table 2"* or *"Set up the environment for my RTX 4090"*

## Environment Setup

We provide two environment configurations:

| Environment             | CUDA  | PyTorch | Use Case                              |
| ----------------------- | ----- | ------- | ------------------------------------- |
| `environment.yml`       | 11.8  | 2.1.2   | Standard GPUs (V100, A100, L40)       |
| `environment_cu124.yml` | 12.4+ | 2.5.1   | Blackwell architecture (RTX PRO 6000) |

**Installation:**

```bash
# Standard installation (CUDA 11.8)
conda env create -f environment.yml
conda activate mtlora
pip install -e ./peft

# For Blackwell GPUs (CUDA 12.4+)
conda env create -f environment_cu124.yml
conda activate mtlora
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e ./peft
```

## Data Preparation

### Included Data (No Setup Required)
- `data/bbh/` - Big Bench Hard evaluation data (27 tasks)
- `data/mmlu_dataset/` - MMLU evaluation data (57 subjects)

### Required Downloads

**1. Base Models (HuggingFace)**
```bash
ln -s /path/to/llama-2-7b ./data/llama-2-7b
ln -s /path/to/llama-2-13b ./data/llama-2-13b  # For Table S7 only
```

**2. Training Datasets**

| Setup | Training Data  | Evaluation  | Size   |
| ----- | -------------- | ----------- | ------ |
| BBH   | Flan-v2 subset | BBH 3-shot  | ~222MB |
| MMLU  | Dolly-15K      | MMLU 5-shot | ~100MB |

```bash
# BBH setup: Flan-v2 subset
mkdir -p data/flan_v2_subset
# Download from: https://huggingface.co/datasets/Muennighoff/flan

# MMLU setup: Dolly-15K
mkdir -p data/dolly-15k-converted
# Download from: https://huggingface.co/datasets/databricks/databricks-dolly-15k
# Convert to instruction format with train.json and validation.json
```

## Reproduce Paper Results

### Main Tables

| Script                                 | Paper Table | Description                     |
| -------------------------------------- | ----------- | ------------------------------- |
| `bash tables/0_main_ablation.sh`       | Table 2     | Contribution of each key design |
| `bash tables/1_routing_granularity.sh` | Table 3     | Routing granularity ablation    |
| `bash tables/2_block_level.sh`         | Table 4     | Block-level adaptation ablation |
| `bash tables/3_llama13b.sh`            | Table S7    | LLaMA-2-13B scalability         |

Each table script runs both BBH and MMLU experiments.

### Analysis Figures

Located in `tables/analysis/`:
- `fig1a_routing_entropy.ipynb` - Figure 1(A): Regularization-routing trade-off
- `fig1b_spectral_conflict.ipynb` - Figure 1(B): Spectral conflict analysis
- `figS2_sv_spectrum.py` - Figure S2: SV spectrum visualization
- `figS3_gradient_perlayer.py` - Figure S3: Per-layer gradient correlation
- `figS4_routing_pattern.py` - Figure S4: Routing weight patterns

## Experiment Script Parameters

All experiments use `train.py` with the following key parameters:

### Method Selection
```bash
--method lora                # Standard single LoRA
--method hydralora           # HydraLoRA baseline (multi-expert, no mtLoRA extensions)
--method mtlora              # mtLoRA (enables block adapter, spectral reg, FGR options)
```

### mtLoRA Components

**Block-Level Adaptation:**
```bash
--enable_block_adapter       # Enable block-level instead of component-level
--block_adapter_type ffn     # Options: attention, ffn, both
--block_adapter_style lowrank
```

**Spectral-Aware Regularization:**
```bash
--enable_spectral_reg        # Enable spectral regularization
--spectral_reg_lambda 1.0    # Regularization strength
--spectral_reg_frequency 1   # SVD frequency (per epoch)
```

**Fine-Grained Routing:**
```bash
--enable_fine_grained_routing
--routing_group_size 2048    # Smaller = finer granularity (more params)
```

### Common Hyperparameters
```bash
--lora_rank 16               # LoRA rank
--lora_alpha 64              # LoRA alpha scaling
--learning_rate 0.0002       # Learning rate
--per_device_train_batch_size 16
--num_train_epochs 1
--max_seq_length 512
```

## Output Structure

After running experiments:
```
output/
├── bbh/                     # BBH experiments (train Flan-v2, eval BBH)
│   ├── 0_baseline/
│   │   ├── sft_lora_model/  # Trained adapter
│   │   └── bbh_eval/        # Evaluation results
│   └── block/ffn/
├── mmlu/                    # MMLU experiments (train Dolly-15K, eval MMLU)
│   ├── 0_baseline/
│   │   ├── sft_lora_model/
│   │   └── mmlu_5shot/
│   └── block/ffn/
└── llama13/                 # LLaMA-13B experiments

logs/                        # Training logs (TensorBoard compatible)
```

## Custom Experiments

### BBH Setup (Train on Flan-v2, Eval on BBH)

```bash
python train.py \
    --method mtlora \
    --model_name_or_path ./data/llama-2-7b \
    --dataset_dir ./data/flan_v2_subset \
    --output_dir ./output/custom_bbh \
    --lora_rank 16 --lora_nums 16 --enable_blc \
    --enable_block_adapter --block_adapter_type ffn \
    --enable_spectral_reg --spectral_reg_lambda 1.0 \
    --enable_fine_grained_routing --routing_group_size 2048 \
    --bf16 --num_train_epochs 1

python eval_bbh.py \
    --model_name_or_path ./data/llama-2-7b \
    --lora_checkpoint ./output/custom_bbh/sft_lora_model \
    --output_dir ./output/custom_bbh/bbh_eval \
    --num_few_shot 3
```

### MMLU Setup (Train on Dolly-15K, Eval on MMLU)

```bash
python train.py \
    --method mtlora \
    --model_name_or_path ./data/llama-2-7b \
    --dataset_dir ./data/dolly-15k-converted \
    --output_dir ./output/custom_mmlu \
    --lora_rank 16 --lora_nums 16 --enable_blc \
    --enable_block_adapter --block_adapter_type ffn \
    --enable_spectral_reg --spectral_reg_lambda 0.5 \
    --enable_fine_grained_routing --routing_group_size 2048 \
    --bf16 --num_train_epochs 1

python eval_mmlu.py \
    --model_name_or_path ./data/llama-2-7b \
    --lora_checkpoint ./output/custom_mmlu/sft_lora_model \
    --output_dir ./output/custom_mmlu/mmlu_5shot \
    --num_few_shot 5 \
    --mmlu_data_dir ./data/mmlu_dataset
```

## Hardware Requirements

| Experiment            | GPU Memory | Recommended  |
| --------------------- | ---------- | ------------ |
| LLaMA-7B (single GPU) | ~24GB      | RTX PRO 6000 |
| LLaMA-7B (DDP, 2 GPU) | ~16GB each | 2x L40       |
| LLaMA-13B             | ~48GB      | A100-80GB    |

For memory-constrained setups, reduce `--per_device_train_batch_size` and increase `--gradient_accumulation_steps`.

## License

This project is licensed under the Apache License 2.0.
