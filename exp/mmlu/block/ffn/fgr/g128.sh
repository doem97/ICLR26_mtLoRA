#!/bin/bash
# FGR Granularity (MMLU): Block-Level, group_size=128
# Dataset: Dolly-15K -> MMLU Evaluation

set -e

echo "========================================"
echo "[2b] Block FGR: group_size=128 (MMLU)"
echo "========================================"

export CUDA_VISIBLE_DEVICES="0"
export NPROC_PER_NODE=1

MODEL_PATH="./data/llama-2-7b"
DATASET_DIR="./data/dolly-15k-converted"
VALIDATION_FILE="./data/dolly-15k-converted/validation.json"
OUTPUT_DIR="./output/mmlu/2_fgr_granularity/2b_block/g128"
LOG_DIR="./logs/mmlu/2_fgr_granularity/2b_block/g128"

LR=0.0002
LORA_RANK=16
LORA_ALPHA=64
LORA_NUMS=16
LORA_DROPOUT=0.05
BATCH_SIZE=16
GRAD_ACCUM_STEPS=2
NUM_EPOCHS=1
MAX_SEQ_LENGTH=512

BLOCK_ADAPTER_TYPE="ffn"
BLOCK_ADAPTER_STYLE="lowrank"
BLOCK_ADAPTER_RANK=16
ROUTING_GROUP_SIZE=128

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
    --lora_dropout $LORA_DROPOUT \
    --load_in_kbits 16 --gradient_checkpointing \
    --overwrite_output_dir --logging_first_step True \
    --enable_blc --enable_block_adapter \
    --block_adapter_type $BLOCK_ADAPTER_TYPE \
    --block_adapter_style $BLOCK_ADAPTER_STYLE \
    --block_adapter_rank $BLOCK_ADAPTER_RANK \
    --enable_fine_grained_routing \
    --routing_group_size $ROUTING_GROUP_SIZE

python eval_mmlu.py \
  --model_name_or_path "$MODEL_PATH" \
  --lora_checkpoint "$OUTPUT_DIR/sft_lora_model" \
  --output_dir "$OUTPUT_DIR/mmlu_5shot" \
  --logging_dir "$LOG_DIR/mmlu_5shot" \
  --num_few_shot 5 --batch_size $BATCH_SIZE --num_gpus 1 \
  --auto_batch --adaptive_max_length --fallback_batch_size 1 \
  --mmlu_data_dir "./data/mmlu_dataset"

