#!/bin/bash
# Table 4: Ablation of Block-Level Adaptation
# BBH (2 GPU) and MMLU (1 GPU) experiments
set -e
cd "$(dirname "$0")/.."

echo "[Table 4] Block-Level Adaptation - BBH"
# Component-level baseline (q_proj, v_proj)
bash exp/bbh/component/0_baseline.sh
# Block Attention only
bash exp/bbh/block/position/attn.sh
# Block FFN only
bash exp/bbh/block/position/ffn.sh
# Block Both (Attention + FFN)
bash exp/bbh/block/position/both.sh

echo "[Table 4] Block-Level Adaptation - MMLU"
# Component-level baseline
bash exp/mmlu/component/0_baseline.sh
# Block Attention only
bash exp/mmlu/block/position/attn.sh
# Block FFN only
bash exp/mmlu/block/position/ffn.sh
# Block Both
bash exp/mmlu/block/position/both.sh

echo "Done. Results in output/bbh/ and output/mmlu/"
