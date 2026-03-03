#!/bin/bash
# Table S7: Scalability to LLaMA-2-13B on MMLU
# Requires: LLaMA-2-13B model, single GPU with ~80GB VRAM
set -e
cd "$(dirname "$0")/.."

if [ ! -d "./data/llama-2-13b" ]; then
    echo "ERROR: LLaMA-2-13B not found at ./data/llama-2-13b"
    exit 1
fi

echo "[Table S7] LLaMA-2-13B Scalability"
# Baseline: HydraLoRA
bash exp/llama13/0_baseline.sh
# Block FFN only
bash exp/llama13/1_block_ffn.sh
# Block FFN + Fine-grained routing
bash exp/llama13/2_fgr_g2048.sh

echo "Done. Results in output/llama13/"
