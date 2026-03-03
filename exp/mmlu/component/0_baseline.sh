#!/bin/bash
# MMLU Ablation: Baseline (HydraLoRA)
# Dataset: Dolly-15K -> MMLU Evaluation
#
# Baseline for all ablation experiments in this folder:
# - No block adapter
# - No fine-grained routing (scalar routing)
# - No spectral regularization
# - Only HydraLoRA on q_proj, v_proj with BLC

set -e

echo "========================================"
echo "[0] Baseline: HydraLoRA (MMLU)"
echo "========================================"

export CUDA_VISIBLE_DEVICES="0"
export NPROC_PER_NODE=1

MODEL_PATH="./data/llama-2-7b"
TOKENIZER_PATH="./data/llama-2-7b"
DATASET_DIR="./data/dolly-15k-converted"
VALIDATION_FILE="./data/dolly-15k-converted/validation.json"
OUTPUT_DIR="./output/mmlu/0_baseline"
LOG_DIR="./logs/mmlu/0_baseline"

LR=0.0002
LORA_RANK=16
LORA_ALPHA=64
LORA_NUMS=16
LORA_DROPOUT=0.05
TRAINABLE_MODULES="q_proj,v_proj"
BATCH_SIZE=16
GRAD_ACCUM_STEPS=2
NUM_EPOCHS=1
MAX_SEQ_LENGTH=512

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Dataset: $DATASET_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  LoRA: rank=$LORA_RANK, alpha=$LORA_ALPHA, experts=$LORA_NUMS"
echo "  Training: lr=$LR, batch=$BATCH_SIZE x $GRAD_ACCUM_STEPS"

python train.py \
    --method hydralora \
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
    --gradient_checkpointing \
    --overwrite_output_dir \
    --logging_first_step True \
    --enable_blc

CHECKPOINT_DIR="$OUTPUT_DIR/sft_lora_model"

python eval_mmlu.py \
  --model_name_or_path "$MODEL_PATH" \
  --lora_checkpoint "$CHECKPOINT_DIR" \
  --output_dir "$OUTPUT_DIR/mmlu_5shot" \
  --logging_dir "$LOG_DIR/mmlu_5shot" \
  --num_few_shot 5 \
  --batch_size $BATCH_SIZE \
  --num_gpus "$NPROC_PER_NODE" \
  --auto_batch \
  --adaptive_max_length \
  --fallback_batch_size 1 \
  --mmlu_data_dir "./data/mmlu_dataset"

echo "========================================"
echo "Baseline Completed!"
echo "========================================"
echo "Training Output: $OUTPUT_DIR"
echo "Evaluation Results: $OUTPUT_DIR/mmlu_5shot"

