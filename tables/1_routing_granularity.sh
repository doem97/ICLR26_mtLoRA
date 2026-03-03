#!/bin/bash
# Table 3: Ablation of Routing Granularity
# BBH (2 GPU) and MMLU (1 GPU) experiments
set -e
cd "$(dirname "$0")/.."

echo "[Table 3] Routing Granularity - BBH"
# g=4096 (scalar routing, equivalent to HydraLoRA)
bash exp/bbh/block/ffn/fgr/g4096.sh
# g=2048 (2 groups per dimension)
bash exp/bbh/block/ffn/fgr/g2048.sh
# g=512 (8 groups)
bash exp/bbh/block/ffn/fgr/g512.sh
# g=256 (16 groups)
bash exp/bbh/block/ffn/fgr/g256.sh
# g=128 (32 groups, finest granularity)
bash exp/bbh/block/ffn/fgr/g128.sh

echo "[Table 3] Routing Granularity - MMLU"
# g=4096 (scalar routing)
bash exp/mmlu/block/ffn/fgr/g4096.sh
# g=2048
bash exp/mmlu/block/ffn/fgr/g2048.sh
# g=512
bash exp/mmlu/block/ffn/fgr/g512.sh
# g=256
bash exp/mmlu/block/ffn/fgr/g256.sh
# g=128
bash exp/mmlu/block/ffn/fgr/g128.sh

echo "Done. Results in output/bbh/block/ffn/fgr/ and output/mmlu/block/ffn/fgr/"
