#!/bin/bash
# FGR Granularity (MMLU): Block-Level, group_size=2048
set -e
echo "[2b] Block FGR: group_size=2048 (MMLU)"
export CUDA_VISIBLE_DEVICES="0"
MODEL_PATH="./data/llama-2-7b"
DATASET_DIR="./data/dolly-15k-converted"
OUTPUT_DIR="./output/mmlu/2_fgr_granularity/2b_block/g2048"
LOG_DIR="./logs/mmlu/2_fgr_granularity/2b_block/g2048"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
python train.py --method mtlora --model_name_or_path "$MODEL_PATH" --tokenizer_name_or_path "$MODEL_PATH" \
    --dataset_dir "$DATASET_DIR" --validation_file "$DATASET_DIR/validation.json" \
    --per_device_train_batch_size 16 --do_train --seed 42 --bf16 --num_train_epochs 1 --learning_rate 0.0002 \
    --gradient_accumulation_steps 2 --logging_strategy steps --logging_steps 10 --save_strategy epoch --save_total_limit 1 \
    --max_seq_length 512 --output_dir "$OUTPUT_DIR" --logging_dir "$LOG_DIR" \
    --lora_rank 16 --lora_alpha 64 --lora_nums 16 --lora_dropout 0.05 \
    --load_in_kbits 16 --gradient_checkpointing --overwrite_output_dir --logging_first_step True \
    --enable_blc --enable_block_adapter --block_adapter_type ffn --block_adapter_style lowrank --block_adapter_rank 16 \
    --enable_fine_grained_routing --routing_group_size 2048
python eval_mmlu.py --model_name_or_path "$MODEL_PATH" --lora_checkpoint "$OUTPUT_DIR/sft_lora_model" \
  --output_dir "$OUTPUT_DIR/mmlu_5shot" --logging_dir "$LOG_DIR/mmlu_5shot" --num_few_shot 5 --batch_size 16 --num_gpus 1 \
  --auto_batch --adaptive_max_length --fallback_batch_size 1 --mmlu_data_dir "./data/mmlu_dataset"

