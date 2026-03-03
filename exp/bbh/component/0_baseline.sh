#!/bin/bash
# BBH Ablation: Baseline (HydraLoRA)
# Dataset: Flan V2 Subset -> BBH Evaluation
# 
# Baseline for all ablation experiments:
# - HydraLoRA on q_proj, v_proj with BLC
# - No block adapter
# - No fine-grained routing
# - No spectral regularization

set -e

echo "========================================"
echo "[0] Baseline: HydraLoRA"
echo "========================================"
echo "Features: HydraLoRA + BLC only"
echo "No mtLoRA extensions"
echo "========================================"

export CUDA_VISIBLE_DEVICES="0,1"
export NPROC_PER_NODE=2

# Model and data paths
MODEL_PATH="./data/llama-2-7b"
TOKENIZER_PATH="./data/llama-2-7b"
DATASET_DIR="./data/flan_v2_subset"
VALIDATION_FILE="./data/flan_v2_subset/train.json"
OUTPUT_DIR="./output/bbh/0_baseline"
LOG_DIR="./logs/bbh/0_baseline"

# Shared hyperparameters (same as other ablations)
LR=0.0002
LORA_RANK=16
LORA_ALPHA=64
LORA_NUMS=16
LORA_DROPOUT=0.05
TRAINABLE_MODULES="q_proj,v_proj"
BATCH_SIZE=16
GRAD_ACCUM_STEPS=1
NUM_EPOCHS=1
MAX_SEQ_LENGTH=512

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  LoRA: rank=$LORA_RANK, alpha=$LORA_ALPHA, experts=$LORA_NUMS"
echo "  Trainable: $TRAINABLE_MODULES"

accelerate launch --num_processes $NPROC_PER_NODE --main_process_port 29600 \
    train.py \
    --deepspeed ./configs/ds_zero2_multi_gpu.json \
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
    --ddp_find_unused_parameters False \
    --gradient_checkpointing \
    --overwrite_output_dir \
    --logging_first_step True \
    --enable_blc

echo "========================================"
echo "Training Completed!"
echo "========================================"

# Find checkpoint
CHECKPOINT_STEP=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
if [ -z "$CHECKPOINT_STEP" ]; then
  CHECKPOINT_DIR="$OUTPUT_DIR/sft_lora_model"
else
  CHECKPOINT_DIR="$OUTPUT_DIR/checkpoint-$CHECKPOINT_STEP"
fi

echo "========================================"
echo "BBH Evaluation (3-shot)"
echo "========================================"

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

