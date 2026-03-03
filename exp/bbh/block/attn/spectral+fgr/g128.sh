#!/bin/bash
# Block Attention + Spectral (λ=1.0) + FGR g=4096
# Dataset: Flan V2 Subset -> BBH Evaluation
# Based on: spectral/lambda/1.0.sh (best BBH config)

set -e

# ============================================================
# KEY VARIABLES
# ============================================================
SPECTRAL_REG_LAMBDA=1.0
SPECTRAL_REG_STEPS=500  # 1x/epoch
ROUTING_GROUP_SIZE=128
# ============================================================

OUTPUT_DIR="./output/bbh/block/attn/spectral+fgr/bbh/g128"
LOG_DIR="./logs/bbh/block/attn/spectral+fgr/bbh/g128"

echo "========================================"
echo "[Block Attn + Spectral + FGR] λ=$SPECTRAL_REG_LAMBDA, g=$ROUTING_GROUP_SIZE"
echo "========================================"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE=2

MODEL_PATH="./data/llama-2-7b"
TOKENIZER_PATH="./data/llama-2-7b"
DATASET_DIR="./data/flan_v2_subset"
VALIDATION_FILE="./data/flan_v2_subset/train.json"

LR=0.0002
LORA_RANK=16
LORA_ALPHA=64
LORA_NUMS=16
LORA_DROPOUT=0.05
BATCH_SIZE=8
GRAD_ACCUM_STEPS=2
NUM_EPOCHS=1
MAX_SEQ_LENGTH=512

BLOCK_ADAPTER_TYPE="attention"
BLOCK_ADAPTER_STYLE="lowrank"
BLOCK_ADAPTER_RANK=16

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

accelerate launch --multi_gpu --num_processes $NPROC_PER_NODE --main_process_port 29768 \
    train.py \
    --method mtlora \
    --model_name_or_path "$MODEL_PATH" \
    --tokenizer_name_or_path "$TOKENIZER_PATH" \
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
    --ddp_find_unused_parameters False \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --logging_first_step True \
    --enable_blc \
    --enable_block_adapter \
    --block_adapter_type $BLOCK_ADAPTER_TYPE \
    --block_adapter_style $BLOCK_ADAPTER_STYLE \
    --block_adapter_rank $BLOCK_ADAPTER_RANK \
    --enable_spectral_reg \
    --spectral_reg_lambda $SPECTRAL_REG_LAMBDA \
    --spectral_reg_steps $SPECTRAL_REG_STEPS \
    --enable_fine_grained_routing \
    --routing_group_size $ROUTING_GROUP_SIZE

CHECKPOINT_STEP=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
if [ -z "$CHECKPOINT_STEP" ]; then
  CHECKPOINT_DIR="$OUTPUT_DIR/sft_lora_model"
else
  CHECKPOINT_DIR="$OUTPUT_DIR/checkpoint-$CHECKPOINT_STEP"
fi

python eval_bbh.py \
  --model_name_or_path "$MODEL_PATH" \
  --lora_checkpoint "$CHECKPOINT_DIR" \
  --output_dir "$OUTPUT_DIR/bbh_eval" \
  --logging_dir "$LOG_DIR/bbh_eval" \
  --batch_size $BATCH_SIZE \
  --num_gpus "$NPROC_PER_NODE" \
  --num_few_shot 3 \
  --bbh_data_dir "./data/bbh" \
  --auto_batch \
  --fallback_batch_size 1

echo "Done: $OUTPUT_DIR"

