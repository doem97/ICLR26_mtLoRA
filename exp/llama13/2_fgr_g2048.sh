#!/bin/bash
# Llama2-13B MMLU Ablation: +Block FFN +FGR g=2560
# Dataset: Dolly-15K -> MMLU Evaluation
#
# Config: HydraLoRA + Block FFN + Fine-Grained Routing (g=2560)
# Delta from 1_block_ffn: +enable_fine_grained_routing, +routing_group_size=2560
# Note: 13B hidden_size=5120, so g=2560 gives 2 groups (equivalent to 7B's g=2048)

set -e

# ============================================================
# KEY VARIABLES
# ============================================================
MODEL_PATH="./data/llama-2-13b"
BLOCK_ADAPTER_TYPE="ffn"
BLOCK_ADAPTER_STYLE="lowrank"
BLOCK_ADAPTER_RANK=16
ROUTING_GROUP_SIZE=2560  # 5120/2560=2 groups (7B uses 4096/2048=2)
# ============================================================

OUTPUT_DIR="./output/llama13/2_fgr_g2560"
LOG_DIR="./logs/llama13/2_fgr_g2560"

echo "========================================"
echo "[Llama2-13B] +Block FFN +FGR g=$ROUTING_GROUP_SIZE (MMLU)"
echo "========================================"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_DIR="./data/dolly-15k-converted"
VALIDATION_FILE="./data/dolly-15k-converted/validation.json"

LR=0.0002
LORA_RANK=16
LORA_ALPHA=64
LORA_NUMS=16
LORA_DROPOUT=0.05
BATCH_SIZE=32
GRAD_ACCUM_STEPS=1
NUM_EPOCHS=1
MAX_SEQ_LENGTH=512

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

python train.py \
    --method mtlora \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name_or_path "$MODEL_PATH" \
    --dataset_dir "$DATASET_DIR" \
    --validation_file "$VALIDATION_FILE" \
    --per_device_train_batch_size $BATCH_SIZE \
    --do_train \
    --seed 42 \
    --bf16 \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LR \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy epoch \
    --save_total_limit 1 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir "$OUTPUT_DIR" \
    --logging_dir "$LOG_DIR" \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_nums $LORA_NUMS \
    --lora_dropout $LORA_DROPOUT \
    --load_in_kbits 16 \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --logging_first_step True \
    --enable_blc \
    --enable_block_adapter \
    --block_adapter_type $BLOCK_ADAPTER_TYPE \
    --block_adapter_style $BLOCK_ADAPTER_STYLE \
    --block_adapter_rank $BLOCK_ADAPTER_RANK \
    --enable_fine_grained_routing \
    --routing_group_size $ROUTING_GROUP_SIZE

CHECKPOINT_STEP=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
if [ -z "$CHECKPOINT_STEP" ]; then
  CHECKPOINT_DIR="$OUTPUT_DIR/sft_lora_model"
else
  CHECKPOINT_DIR="$OUTPUT_DIR/checkpoint-$CHECKPOINT_STEP"
fi

python eval_mmlu.py \
  --model_name_or_path "$MODEL_PATH" \
  --lora_checkpoint "$CHECKPOINT_DIR" \
  --output_dir "$OUTPUT_DIR/mmlu_5shot" \
  --logging_dir "$LOG_DIR/mmlu_5shot" \
  --num_few_shot 5 \
  --batch_size 16 \
  --num_gpus 1 \
  --auto_batch \
  --adaptive_max_length \
  --fallback_batch_size 1 \
  --mmlu_data_dir "./data/mmlu_dataset"

echo "Done: $OUTPUT_DIR"

