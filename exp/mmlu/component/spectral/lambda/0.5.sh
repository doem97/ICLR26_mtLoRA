#!/bin/bash
# Spectral Lambda (MMLU): lambda=0.5
# Dataset: Dolly-15K -> MMLU Evaluation

set -e

# ============================================================
# KEY VARIABLES
# ============================================================
SPECTRAL_REG_LAMBDA=0.5
SPECTRAL_REG_STEPS=280  # ~469 steps/epoch, 1x at step 280
# ============================================================

# Output paths
OUTPUT_DIR="./output/mmlu/3_spectral/3a_lambda/lambda_0.5"
LOG_DIR="./logs/mmlu/3_spectral/3a_lambda/lambda_0.5"

echo "========================================"
echo "[3a] Spectral Lambda: $SPECTRAL_REG_LAMBDA (MMLU)"
echo "========================================"

export CUDA_VISIBLE_DEVICES="0"
export NPROC_PER_NODE=1

# Model and data paths
MODEL_PATH="./data/llama-2-7b"
DATASET_DIR="./data/dolly-15k-converted"
VALIDATION_FILE="./data/dolly-15k-converted/validation.json"

# Shared hyperparameters
LR=0.0002
LORA_RANK=16
LORA_ALPHA=64
LORA_NUMS=16
LORA_DROPOUT=0.05
TRAINABLE_MODULES="q_proj,v_proj"
BATCH_SIZE=8
GRAD_ACCUM_STEPS=4
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
    --do_train --seed 42 --bf16 \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LR \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --logging_strategy steps --logging_steps 10 \
    --save_strategy epoch --save_total_limit 1 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --output_dir "$OUTPUT_DIR" --logging_dir "$LOG_DIR" \
    --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --lora_nums $LORA_NUMS \
    --trainable "$TRAINABLE_MODULES" --lora_dropout $LORA_DROPOUT \
    --load_in_kbits 16 --gradient_checkpointing \
    --overwrite_output_dir --logging_first_step True \
    --enable_blc --enable_spectral_reg \
    --spectral_reg_lambda $SPECTRAL_REG_LAMBDA \
    --spectral_reg_steps $SPECTRAL_REG_STEPS

# Dynamic checkpoint detection
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
  --num_few_shot 5 --batch_size $BATCH_SIZE --num_gpus 1 \
  --auto_batch --adaptive_max_length --fallback_batch_size 1 \
  --mmlu_data_dir "./data/mmlu_dataset"
