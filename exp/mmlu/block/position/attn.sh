#!/bin/bash
# Block Position Ablation (MMLU): Attention Only
# Dataset: Dolly-15K -> MMLU Evaluation

set -e

echo "========================================"
echo "[1a] Block Adapter: Attention Only (MMLU)"
echo "========================================"

export CUDA_VISIBLE_DEVICES="0"
export NPROC_PER_NODE=1

MODEL_PATH="./data/llama-2-7b"
TOKENIZER_PATH="./data/llama-2-7b"
DATASET_DIR="./data/dolly-15k-converted"
VALIDATION_FILE="./data/dolly-15k-converted/validation.json"
OUTPUT_DIR="./output/mmlu/1_block_position/1a_attn_only"
LOG_DIR="./logs/mmlu/1_block_position/1a_attn_only"

LR=0.0002
LORA_RANK=16
LORA_ALPHA=64
LORA_NUMS=16
LORA_DROPOUT=0.05
BATCH_SIZE=16
GRAD_ACCUM_STEPS=2
NUM_EPOCHS=1
MAX_SEQ_LENGTH=512

BLOCK_ADAPTER_TYPE="attention"
BLOCK_ADAPTER_STYLE="lowrank"
BLOCK_ADAPTER_RANK=16

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

python train.py \
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
    --gradient_checkpointing \
    --overwrite_output_dir \
    --logging_first_step True \
    --enable_blc \
    --enable_block_adapter \
    --block_adapter_type $BLOCK_ADAPTER_TYPE \
    --block_adapter_style $BLOCK_ADAPTER_STYLE \
    --block_adapter_rank $BLOCK_ADAPTER_RANK

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

echo "Evaluation Results: $OUTPUT_DIR/mmlu_5shot"

