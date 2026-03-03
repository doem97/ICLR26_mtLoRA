#!/bin/bash
# Spectral Frequency Ablation: 4x per epoch, lambda=0.5
# Dataset: Flan V2 Subset -> BBH Evaluation
# 
# BBH: 844 steps/epoch → triggers at 180, 360, 540, 720 (124 steps remaining)

set -e

echo "========================================"
echo "[3b] Spectral Frequency: 4x/epoch"
echo "========================================"
echo "Lambda: 0.5"
echo "Steps: 180 → triggers at 180, 360, 540, 720"
echo "========================================"

export CUDA_VISIBLE_DEVICES="0,1"
export NPROC_PER_NODE=2

MODEL_PATH="./data/llama-2-7b"
TOKENIZER_PATH="./data/llama-2-7b"
DATASET_DIR="./data/flan_v2_subset"
VALIDATION_FILE="./data/flan_v2_subset/train.json"
OUTPUT_DIR="./output/bbh/3_spectral/lambda0.5_4x"
LOG_DIR="./logs/bbh/3_spectral/lambda0.5_4x"

LR=0.0002
LORA_RANK=16
LORA_ALPHA=64
LORA_NUMS=16
LORA_DROPOUT=0.05
TRAINABLE_MODULES="q_proj,v_proj"
BATCH_SIZE=8
GRAD_ACCUM_STEPS=2
NUM_EPOCHS=1
MAX_SEQ_LENGTH=512

# === KEY VARIABLES ===
SPECTRAL_REG_LAMBDA=0.5
SPECTRAL_REG_STEPS=180

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

accelerate launch --multi_gpu --num_processes $NPROC_PER_NODE --main_process_port 29662 \
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
    --trainable "$TRAINABLE_MODULES" \
    --lora_dropout $LORA_DROPOUT \
    --load_in_kbits 16 \
    --ddp_find_unused_parameters False \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --logging_first_step True \
    --enable_blc \
    --enable_spectral_reg \
    --spectral_reg_lambda $SPECTRAL_REG_LAMBDA \
    --spectral_reg_steps $SPECTRAL_REG_STEPS

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

echo "Evaluation Results: $OUTPUT_DIR/bbh_eval"
