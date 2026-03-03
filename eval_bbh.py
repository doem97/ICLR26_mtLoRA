#!/usr/bin/env python
"""
Fast BBH Evaluation with Multi-GPU Parallel Processing
Includes OOM recovery with automatic batch size reduction
"""

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

# BBH Tasks
BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]


@dataclass
class EvalArguments:
    model_name_or_path: str = field(metadata={"help": "Path to base model"})
    lora_checkpoint: str | None = field(default=None, metadata={"help": "Path to LoRA checkpoint"})
    output_dir: str = field(default="./bbh_results", metadata={"help": "Output directory"})
    logging_dir: str | None = field(default=None, metadata={"help": "Optional separate directory for log files"})
    tasks: str | None = field(default=None, metadata={"help": "Comma-separated tasks (default: all)"})
    batch_size: int = field(default=8, metadata={"help": "Batch size per GPU"})
    num_gpus: int = field(default=1, metadata={"help": "Number of GPUs to use"})
    gpu_ids: str | None = field(
        default=None, metadata={"help": "Comma-separated GPU IDs (auto-detect if not specified)"}
    )
    num_few_shot: int = field(default=3, metadata={"help": "Number of few-shot examples (default: 3)"})
    bbh_data_dir: str = field(default="./data/bbh", metadata={"help": "Local BBH dataset directory"})
    auto_batch: bool = field(default=False, metadata={"help": "Enable auto batch size reduction on OOM"})
    fallback_batch_size: int = field(default=1, metadata={"help": "Minimum batch size when OOM occurs"})
    max_length: int = field(default=2048, metadata={"help": "Maximum sequence length for tokenization"})


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_bbh_task(data_dir, task):
    file_path = os.path.join(data_dir, f"{task}.json")
    with open(file_path) as f:
        data = json.load(f)
    return data


def build_prompt(item, few_shot_examples=None):
    prompt = ""
    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += f"Q: {ex['input']}\nA: {ex['target']}\n\n"

    prompt += f"Q: {item['input']}\nA:"
    return prompt


def run_batch_inference(model, tokenizer, batch_prompts, batch_correct, device, max_length):
    """Run inference on a single batch. Returns predictions list."""
    import torch

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    inputs = inputs.to(device)

    # Add task_types for routing-based methods
    if hasattr(model, "peft_config") and model.peft_config:
        if isinstance(model.peft_config, dict):
            active_adapter = next(iter(model.peft_config.values()))
            peft_type = active_adapter.peft_type
        else:
            peft_type = model.peft_config.peft_type

        if peft_type in ["HYDRALORA", "MMOELORA", "MMOELORAS"]:
            batch_size_actual = inputs["input_ids"].shape[0]
            inputs["task_types"] = torch.zeros(batch_size_actual, dtype=torch.long, device=device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    predictions = []
    for j, (output, correct_answer) in enumerate(zip(outputs, batch_correct, strict=True)):
        input_length = inputs["input_ids"][j].shape[0]
        generated = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()

        is_correct = correct_answer.lower() in generated.lower() or generated.lower().startswith(
            correct_answer.lower()
        )

        predictions.append(
            {
                "question": batch_prompts[j],
                "correct_answer": correct_answer,
                "predicted_answer": generated,
                "correct": is_correct,
            }
        )

    return predictions


def evaluate_task_with_oom_recovery(
    model,
    tokenizer,
    prompts,
    correct_answers,
    device,
    initial_batch_size,
    max_length,
    auto_batch_enabled,
    fallback_batch_size,
    task_name,
    gpu_id,
):
    """Evaluate a single task with OOM recovery."""
    import torch
    from tqdm import tqdm

    predictions = []
    current_batch_size = initial_batch_size
    current_max_length = max_length
    i = 0

    with torch.no_grad():
        pbar = tqdm(total=len(prompts), desc=f"Eval {task_name}", leave=False)

        while i < len(prompts):
            batch_prompts = prompts[i : i + current_batch_size]
            batch_correct = correct_answers[i : i + current_batch_size]

            try:
                batch_predictions = run_batch_inference(
                    model, tokenizer, batch_prompts, batch_correct, device, current_max_length
                )
                predictions.extend(batch_predictions)
                pbar.update(len(batch_prompts))
                i += current_batch_size

            except torch.cuda.OutOfMemoryError as e:
                if not auto_batch_enabled:
                    # Re-raise if auto_batch is disabled
                    raise

                logger.warning(f"GPU {gpu_id}: OOM on task {task_name}: {e}")
                torch.cuda.empty_cache()

                # Try reducing batch size first
                if current_batch_size > fallback_batch_size:
                    new_batch_size = max(fallback_batch_size, current_batch_size // 2)
                    logger.warning(f"GPU {gpu_id}: Reducing batch_size {current_batch_size} -> {new_batch_size}")
                    current_batch_size = new_batch_size
                # Then try reducing max_length
                elif current_max_length > 1024:
                    new_max_length = max(1024, current_max_length - 256)
                    logger.warning(f"GPU {gpu_id}: Reducing max_length {current_max_length} -> {new_max_length}")
                    current_max_length = new_max_length
                else:
                    # Already at minimum settings, skip this sample
                    logger.error(f"GPU {gpu_id}: Cannot reduce further, skipping sample {i} in task {task_name}")
                    predictions.append(
                        {
                            "question": batch_prompts[0] if batch_prompts else "",
                            "correct_answer": batch_correct[0] if batch_correct else "",
                            "predicted_answer": "[OOM_SKIPPED]",
                            "correct": False,
                        }
                    )
                    pbar.update(1)
                    i += 1

        pbar.close()

    return predictions


def evaluate_tasks_on_gpu(
    gpu_id: int,
    physical_gpu_id: int,
    tasks: list[str],
    model_path: str,
    lora_path: str | None,
    batch_size: int,
    num_few_shot: int,
    bbh_data_dir: str,
    auto_batch: bool = False,
    fallback_batch_size: int = 1,
    max_length: int = 2048,
) -> dict:
    """Evaluate tasks on a single GPU with OOM recovery"""

    # Import torch here after setting CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)

    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM

    from peft import PeftModel

    # Apply HydraLoRA patches in each subprocess
    try:
        from peft.utils.transformers_patch import patch_llama_for_hydralora

        patch_llama_for_hydralora()
        logger.info(f"GPU {gpu_id}: HydraLoRA patches applied")
    except ImportError:
        pass

    device = "cuda:0"
    torch.cuda.set_device(0)

    logger.info(f"GPU {gpu_id}: Loading model for {len(tasks)} tasks")
    if auto_batch:
        logger.info(f"GPU {gpu_id}: Auto batch enabled (fallback_batch_size={fallback_batch_size})")

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_path:
        logger.info(f"GPU {gpu_id}: Loading LoRA from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.bfloat16()
        model = model.to(device)

        if hasattr(model, "peft_config") and model.peft_config:
            if isinstance(model.peft_config, dict):
                peft_type = next(iter(model.peft_config.values())).peft_type
            else:
                peft_type = model.peft_config.peft_type
            logger.info(f"GPU {gpu_id}: Detected PEFT type: {peft_type}")

    model.eval()
    results = {}

    for task in tasks:
        logger.info(f"GPU {gpu_id}: Evaluating {task}")

        try:
            dataset = load_bbh_task(bbh_data_dir, task)
        except FileNotFoundError:
            logger.warning(f"GPU {gpu_id}: Task {task} file not found in {bbh_data_dir}. Skipping.")
            continue

        # Use first N examples as few-shot, evaluate on the rest
        if num_few_shot > 0:
            few_shot_examples = dataset[:num_few_shot]
            eval_dataset = dataset[num_few_shot:]
        else:
            few_shot_examples = []
            eval_dataset = dataset

        prompts = []
        correct_answers = []

        for item in eval_dataset:
            prompt = build_prompt(item, few_shot_examples)
            prompts.append(prompt)
            correct_answers.append(item["target"])

        # Evaluate with OOM recovery
        try:
            predictions = evaluate_task_with_oom_recovery(
                model=model,
                tokenizer=tokenizer,
                prompts=prompts,
                correct_answers=correct_answers,
                device=device,
                initial_batch_size=batch_size,
                max_length=max_length,
                auto_batch_enabled=auto_batch,
                fallback_batch_size=fallback_batch_size,
                task_name=task,
                gpu_id=gpu_id,
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU {gpu_id}: Fatal OOM on task {task}, skipping entire task: {e}")
            torch.cuda.empty_cache()
            continue

        correct_count = sum(1 for p in predictions if p["correct"])
        accuracy = correct_count / len(predictions) if predictions else 0

        results[task] = {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": len(predictions),
            "predictions": predictions,
        }

        logger.info(f"GPU {gpu_id}: {task} = {accuracy:.2%} ({correct_count}/{len(predictions)})")

    return results


def main():
    from transformers import HfArgumentParser

    # Apply HydraLoRA patches
    try:
        from peft.utils.transformers_patch import patch_llama_for_hydralora

        patch_llama_for_hydralora()
    except ImportError:
        pass

    # Set multiprocessing start method to spawn to avoid CUDA context issues
    import multiprocessing

    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    parser = HfArgumentParser(EvalArguments)
    args = parser.parse_args_into_dataclasses()[0]

    os.makedirs(args.output_dir, exist_ok=True)

    if args.logging_dir:
        os.makedirs(args.logging_dir, exist_ok=True)
        log_file = os.path.join(args.logging_dir, "evaluation.log")
    else:
        log_file = os.path.join(args.output_dir, "evaluation.log")

    # Setup logging
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    logger.info("Starting BBH evaluation")
    if args.auto_batch:
        logger.info(f"Auto batch enabled: will reduce batch size on OOM (min={args.fallback_batch_size})")

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
    else:
        tasks = BBH_TASKS

    # GPU Setup - handle CUDA_VISIBLE_DEVICES mapping
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if args.gpu_ids:
        logical_gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    else:
        if visible_devices:
            num_visible = len(visible_devices.split(","))
            logical_gpu_ids = list(range(min(args.num_gpus, num_visible)))
        else:
            logical_gpu_ids = list(range(args.num_gpus))

    # Map logical GPU IDs to physical GPU IDs
    if visible_devices:
        physical_gpu_list = [int(x.strip()) for x in visible_devices.split(",")]
        physical_gpu_ids = [physical_gpu_list[i] for i in logical_gpu_ids]
    else:
        physical_gpu_ids = logical_gpu_ids

    num_gpus = len(logical_gpu_ids)

    logger.info(f"Logical GPU IDs: {logical_gpu_ids}")
    logger.info(f"Physical GPU IDs: {physical_gpu_ids}")

    # Distribute tasks
    tasks_per_gpu = len(tasks) // num_gpus
    extra_tasks = len(tasks) % num_gpus
    distributed_tasks = []
    start_idx = 0
    for i in range(num_gpus):
        end_idx = start_idx + tasks_per_gpu + (1 if i < extra_tasks else 0)
        distributed_tasks.append(tasks[start_idx:end_idx])
        start_idx = end_idx

    logger.info(f"Evaluating {len(tasks)} tasks on {num_gpus} GPUs")

    all_results = {}
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []
        for i, gpu_tasks in enumerate(distributed_tasks):
            if gpu_tasks:
                future = executor.submit(
                    evaluate_tasks_on_gpu,
                    logical_gpu_ids[i],
                    physical_gpu_ids[i],
                    gpu_tasks,
                    args.model_name_or_path,
                    args.lora_checkpoint,
                    args.batch_size,
                    args.num_few_shot,
                    args.bbh_data_dir,
                    args.auto_batch,
                    args.fallback_batch_size,
                    args.max_length,
                )
                futures.append(future)

        for future in futures:
            try:
                all_results.update(future.result())
            except Exception as e:
                logger.error(f"GPU evaluation failed: {e}")

    total_time = time.time() - start_time

    # Summary
    if all_results:
        avg_accuracy = sum(r["accuracy"] for r in all_results.values()) / len(all_results)
        total_samples = sum(r["total"] for r in all_results.values())
        total_correct = sum(r["correct"] for r in all_results.values())

        results_file = Path(args.output_dir) / "bbh_results.json"
        final_data = {
            "average_accuracy": avg_accuracy,
            "total_correct": total_correct,
            "total_samples": total_samples,
            "num_tasks": len(all_results),
            "total_time": total_time,
            "config": str(args),
            "results": all_results,
        }
        with open(results_file, "w") as f:
            json.dump(final_data, f, indent=2)

        logger.info("=" * 70)
        logger.info("BBH EVALUATION RESULTS")
        logger.info("=" * 70)
        logger.info(f"Average BBH Accuracy: {avg_accuracy:.2%}")
        logger.info(f"Total: {total_correct}/{total_samples}")
        logger.info(f"Total Time: {total_time:.1f}s")
        logger.info(f"GPUs Used: {num_gpus} ({logical_gpu_ids})")
        logger.info(f"Few-shot: {args.num_few_shot}-shot")
        if args.auto_batch:
            logger.info(f"Auto batch: enabled (fallback={args.fallback_batch_size})")
        logger.info("")
        logger.info("Task Scores (sorted by accuracy):")
        sorted_tasks = sorted(all_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        for task, result in sorted_tasks:
            logger.info(f"  {task:45s}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
        logger.info("=" * 70)
        logger.info("Evaluation completed successfully")

        print(f"\n{'=' * 70}")
        print("BBH EVALUATION RESULTS")
        print("=" * 70)
        print(f"Average BBH Accuracy: {avg_accuracy:.2%}")
        print(f"Total: {total_correct}/{total_samples}")
        print(f"Total Time: {total_time:.1f}s")
        print(f"GPUs Used: {num_gpus} ({logical_gpu_ids})")
        print(f"Few-shot: {args.num_few_shot}-shot")
        if args.auto_batch:
            print(f"Auto batch: enabled (fallback={args.fallback_batch_size})")
        print(f"\nFull results saved to: {results_file}")
        print("=" * 70)


if __name__ == "__main__":
    main()
