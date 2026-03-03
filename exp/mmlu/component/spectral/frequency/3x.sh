#!/bin/bash
# Spectral Frequency Ablation: 3x per epoch, lambda=0.5
# Dataset: Dolly-15K -> MMLU Evaluation
# 
# MMLU: 469 steps/epoch → triggers at 130, 260, 390 (79 steps remaining)

set -e

echo "========================================"
echo "[3b] Spectral Frequency: 3x/epoch (MMLU)"
echo "========================================"
echo "Lambda: 0.5"
echo "Steps: 130 → triggers at 130, 260, 390"
echo "========================================"

export CUDA_VISIBLE_DEVICES="0"
export NPROC_PER_NODE=1

MODEL_PATH="./data/llama-2-7b"
DATASET_DIR="./data/dolly-15k-converted"
VALIDATION_FILE="./data/dolly-15k-converted/validation.json"
OUTPUT_DIR="./output/mmlu/mmlu/3_spectral/lambda0.5_3x"
LOG_DIR="./logs/mmlu/mmlu/3_spectral/lambda0.5_3x"

LR=0.0002
LORA_RANK=16
LORA_ALPHA=64
LORA_NUMS=16
LORA_DROPOUT=0.05
TRAINABLE_MODULES="q_proj,v_proj"
BATCH_SIZE=32
GRAD_ACCUM_STEPS=1
NUM_EPOCHS=1
MAX_SEQ_LENGTH=512

# === KEY VARIABLES ===
SPECTRAL_REG_LAMBDA=0.5
SPECTRAL_REG_STEPS=130

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
