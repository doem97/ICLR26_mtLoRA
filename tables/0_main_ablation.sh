#!/bin/bash
# Table 2: Contribution of Each Key Design
# BBH (2 GPU) and MMLU (1 GPU) experiments
set -e
cd "$(dirname "$0")/.."

echo "[Table 2] Main Ablation - BBH"
# Baseline: HydraLoRA (no mtLoRA extensions)
bash exp/bbh/component/0_baseline.sh
# Block-level adaptation (FFN only)
bash exp/bbh/block/ffn/0_baseline.sh
# Block + Spectral regularization
bash exp/bbh/block/ffn/spectral/lambda/1.0.sh
# Block + Fine-grained routing
bash exp/bbh/block/ffn/fgr/g2048.sh
# Full mtLoRA (Block + Spectral + FGR)
bash exp/bbh/block/ffn/spectral+fgr/g2048.sh

echo "[Table 2] Main Ablation - MMLU"
# Baseline: HydraLoRA (no mtLoRA extensions)
bash exp/mmlu/component/0_baseline.sh
# Block-level adaptation (FFN only)
bash exp/mmlu/block/ffn/0_baseline.sh
# Block + Spectral regularization
bash exp/mmlu/block/ffn/spectral/lambda/0.5.sh
# Block + Fine-grained routing
bash exp/mmlu/block/ffn/fgr/g2048.sh
# Full mtLoRA (Block + Spectral + FGR)
bash exp/mmlu/block/ffn/spectral+fgr/g2048.sh

echo "Done. Results in output/bbh/ and output/mmlu/"
