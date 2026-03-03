# mtLoRA - Claude Agent Instructions

You are helping users reproduce experiments from the mtLoRA paper. This file provides context for assisting with environment setup, data configuration, and experiment reproduction.

## Project Structure

```
mtlora/
├── train.py              # Main training script
├── eval_bbh.py           # BBH evaluation (3-shot)
├── eval_mmlu.py          # MMLU evaluation (5-shot)
├── configs/              # DeepSpeed configs
├── data/
│   ├── bbh/              # BBH eval data (included)
│   ├── mmlu_dataset/     # MMLU eval data (included)
│   ├── flan_v2_subset/   # Training data for BBH (user downloads)
│   ├── dolly-15k-converted/  # Training data for MMLU (user downloads)
│   ├── llama-2-7b        # Symlink to model (user creates)
│   └── llama-2-13b       # Symlink to model (user creates)
├── exp/
│   ├── bbh/              # BBH experiment scripts
│   ├── mmlu/             # MMLU experiment scripts
│   └── llama13/          # LLaMA-13B experiment scripts
├── tables/               # Table reproduction scripts
│   ├── 0_main_ablation.sh      # Table 2
│   ├── 1_routing_granularity.sh # Table 3
│   ├── 2_block_level.sh        # Table 4
│   ├── 3_llama13b.sh           # Table S7
│   └── analysis/               # Figure reproduction
├── peft/                 # Custom PEFT library
└── utils/                # Training utilities
```

## Environment Setup

### Detect User's CUDA Version

Run `nvidia-smi` to check CUDA version, then recommend:

| CUDA Version | Environment File        | Notes                    |
| ------------ | ----------------------- | ------------------------ |
| 11.x         | `environment.yml`       | V100, A100, L40          |
| 12.4+        | `environment_cu124.yml` | Blackwell (RTX PRO 6000) |

### Setup Commands

```bash
# Standard (CUDA 11.x)
conda env create -f environment.yml
conda activate mtlora
pip install -e ./peft

# Blackwell GPUs (CUDA 12.4+)
conda env create -f environment_cu124.yml
conda activate mtlora
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e ./peft
```

## Data Setup

### Check Existing Data

```bash
ls -la data/
```

### Required Symlinks for Models

```bash
# LLaMA-2-7B (required for all experiments)
ln -s /path/to/llama-2-7b ./data/llama-2-7b

# LLaMA-2-13B (only for Table S7)
ln -s /path/to/llama-2-13b ./data/llama-2-13b
```

### Training Data Downloads

**For BBH experiments (Flan-v2 subset, ~222MB):**
- Download from: https://huggingface.co/datasets/Muennighoff/flan
- Place as: `data/flan_v2_subset/train.json`

**For MMLU experiments (Dolly-15K, ~100MB):**
- Download from: https://huggingface.co/datasets/databricks/databricks-dolly-15k
- Convert to instruction format
- Place as: `data/dolly-15k-converted/train.json` and `validation.json`

## Table Reproduction

### Table 2: Contribution of Each Key Design

Tests each mtLoRA component's contribution:
- Baseline (HydraLoRA only)
- Block-level adaptation
- Spectral regularization
- Fine-grained routing
- Full mtLoRA (all combined)

```bash
bash tables/0_main_ablation.sh
```

Runs both BBH (2 GPU) and MMLU (1 GPU) experiments.

### Table 3: Routing Granularity Ablation

Tests different routing group sizes: 4096, 2048, 512, 256, 128

```bash
bash tables/1_routing_granularity.sh
```

### Table 4: Block-Level Adaptation

Compares placement strategies: component-level, block-attention, block-FFN, block-both

```bash
bash tables/2_block_level.sh
```

### Table S7: LLaMA-2-13B Scalability

Validates mtLoRA scales to larger models (requires 80GB GPU)

```bash
bash tables/3_llama13b.sh
```

## Key Parameters

### Method Selection
- `--method hydralora`: Baseline (multi-expert LoRA, no mtLoRA extensions)
- `--method mtlora`: Full mtLoRA (enables block adapter, spectral reg, FGR)

### mtLoRA Components
- `--enable_block_adapter --block_adapter_type ffn`: Block-level adaptation
- `--enable_spectral_reg --spectral_reg_lambda 1.0`: Spectral regularization
- `--enable_fine_grained_routing --routing_group_size 2048`: Fine-grained routing

### Common Settings
- `--lora_rank 16 --lora_alpha 64`: LoRA configuration
- `--lora_nums 16`: Number of experts
- `--enable_blc`: Balance loss coefficient

## Troubleshooting

### OOM Errors
Reduce batch size: `--per_device_train_batch_size 8` or `4`
Increase gradient accumulation: `--gradient_accumulation_steps 2`

### Missing Data
Check symlinks exist: `ls -la data/llama-2-7b`
Verify training data: `ls data/flan_v2_subset/` or `ls data/dolly-15k-converted/`

### CUDA Errors
Verify CUDA version: `nvidia-smi`
Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`

## Example Interactions

**User:** "Help me set up the environment"
- Check their CUDA version with `nvidia-smi`
- Recommend appropriate environment file
- Guide through conda setup and peft installation

**User:** "Reproduce Table 2"
- Verify data/models are set up
- Run `bash tables/0_main_ablation.sh`
- Explain what experiments are running

**User:** "Run just the baseline on BBH"
- Run `bash exp/bbh/component/0_baseline.sh`
- This uses `--method hydralora` (no mtLoRA extensions)

**User:** "What GPUs do I need?"
- LLaMA-7B: Single ~24GB GPU (RTX PRO 6000) or 2x ~16GB (L40)
- LLaMA-13B: Single ~48GB GPU (A100-80GB)

