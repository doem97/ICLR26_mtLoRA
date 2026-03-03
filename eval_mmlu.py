#!/usr/bin/env python
"""
Fast MMLU Evaluation with Multi-GPU Parallel Processing
Each GPU processes different subjects for maximum speedup
"""

import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser, LlamaForCausalLM

from peft import PeftModel

# MMLU subjects (57 total)
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

# Category mapping for results
CATEGORIES = {
    "STEM": ["physics", "chemistry", "biology", "computer_science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social_sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other": ["other", "business", "health"],
}

SUBJECT_TO_CATEGORY = {
    "abstract_algebra": "STEM",
    "anatomy": "other",
    "astronomy": "STEM",
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_medicine": "other",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "econometrics": "social_sciences",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "formal_logic": "humanities",
    "global_facts": "other",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_european_history": "humanities",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_mathematics": "STEM",
    "high_school_microeconomics": "social_sciences",
    "high_school_physics": "STEM",
    "high_school_psychology": "social_sciences",
    "high_school_statistics": "STEM",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "human_aging": "other",
    "human_sexuality": "social_sciences",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "machine_learning": "STEM",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "nutrition": "other",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_accounting": "other",
    "professional_law": "humanities",
    "professional_medicine": "other",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    "virology": "other",
    "world_religions": "humanities",
}


@dataclass
class EvalArguments:
    model_name_or_path: str = field(metadata={"help": "Path to base model"})
    lora_checkpoint: str | None = field(default=None, metadata={"help": "Path to LoRA checkpoint"})
    output_dir: str = field(default="./mmlu_results", metadata={"help": "Output directory"})
    logging_dir: str | None = field(default=None, metadata={"help": "Optional separate directory for log files"})
    subjects: str | None = field(default=None, metadata={"help": "Comma-separated subjects (default: all)"})
    num_samples: int | None = field(default=None, metadata={"help": "Limit samples per subject for testing"})
    batch_size: int = field(default=32, metadata={"help": "Batch size per GPU"})
    num_gpus: int = field(default=4, metadata={"help": "Number of GPUs to use"})
    gpu_ids: str | None = field(
        default=None, metadata={"help": "Comma-separated GPU IDs (auto-detect if not specified)"}
    )
    num_few_shot: int = field(default=5, metadata={"help": "Number of few-shot examples (0-5, standard is 5)"})
    auto_batch: bool = field(default=False, metadata={"help": "Enable auto batch size on OOM"})
    adaptive_max_length: bool = field(default=False, metadata={"help": "Adapt max_length based on num_few_shot"})
    fallback_batch_size: int = field(default=1, metadata={"help": "Minimum batch size when OOM"})
    mmlu_data_dir: str | None = field(
        default=None, metadata={"help": "Local MMLU dataset directory (default: download from HF)"}
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_adaptive_max_length(num_few_shot: int, adaptive_enabled: bool = False) -> int:
    """Get adaptive max_length based on number of few-shot examples"""
    if not adaptive_enabled:
        return 2048

    shot_to_length = {
        0: 768,  # 0-shot: question + choices only
        1: 1024,  # 1-shot: + 1 example
        2: 1280,  # 2-shot: + 2 examples
        3: 1536,  # 3-shot: + 3 examples
        4: 1792,  # 4-shot: + 4 examples
        5: 2048,  # 5-shot: original setting
    }

    return shot_to_length.get(num_few_shot, 2048)


def safe_batch_evaluation(
    model,
    tokenizer,
    prompts: list[str],
    correct_answers: list[str],
    initial_batch_size: int,
    max_length: int,
    num_few_shot: int,
    auto_batch_enabled: bool = False,
    fallback_batch_size: int = 1,
) -> list[dict]:
    """Batch evaluation with OOM recovery"""
    if not auto_batch_enabled:
        # Use original logic without OOM handling
        return run_original_batch_evaluation(
            model, tokenizer, prompts, correct_answers, initial_batch_size, max_length
        )

    current_batch_size = initial_batch_size
    current_max_length = max_length
    retry_count = 0
    max_retries = 3

    while retry_count < max_retries:
        try:
            logger.info(
                f"Attempting batch evaluation with batch_size={current_batch_size}, max_length={current_max_length}"
            )
            return run_original_batch_evaluation(
                model, tokenizer, prompts, correct_answers, current_batch_size, current_max_length
            )
        except torch.cuda.OutOfMemoryError as e:
            logger.warning(f"OOM encountered: {e}")
            torch.cuda.empty_cache()

            if current_batch_size > fallback_batch_size:
                current_batch_size = max(fallback_batch_size, current_batch_size // 2)
                logger.warning(f"Reducing batch_size to {current_batch_size}")
            elif current_max_length > 1024:
                current_max_length = max(1024, current_max_length - 256)
                logger.warning(f"Reducing max_length to {current_max_length}")
            else:
                logger.error("Cannot reduce batch_size or max_length further")
                break

            retry_count += 1

    # Final attempt with most conservative settings
    logger.warning("Using most conservative settings as final attempt")
    torch.cuda.empty_cache()
    return run_original_batch_evaluation(model, tokenizer, prompts, correct_answers, fallback_batch_size, 1024)


def run_original_batch_evaluation(
    model, tokenizer, prompts: list[str], correct_answers: list[str], batch_size: int, max_length: int
) -> list[dict]:
    """Original batch evaluation logic extracted as separate function"""
    predictions = []

    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="Batch evaluation"):
            batch_prompts = prompts[i : i + batch_size]
            batch_correct = correct_answers[i : i + batch_size]

            # Tokenize
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
            )

            # Add task_types ONLY for methods that require it (HydraLoRA, MMOELoRA)
            # Check PEFT config if available
            if hasattr(model, "peft_config") and model.peft_config:
                # Handle both single config and dict of configs
                if isinstance(model.peft_config, dict):
                    # Assuming single active adapter for inference
                    active_adapter = next(iter(model.peft_config.values()))
                    peft_type = active_adapter.peft_type
                else:
                    peft_type = model.peft_config.peft_type

                # Only add task_types for routing-based methods
                if peft_type in ["HYDRALORA", "MMOELORA", "MMOELORAS"]:
                    batch_size_actual = inputs["input_ids"].shape[0]
                    inputs["task_types"] = torch.zeros(batch_size_actual, dtype=torch.long, device=model.device)

            # Generate
            outputs = model.generate(
                **inputs.to(model.device),
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                temperature=None,
                top_p=None,
            )

            # Process outputs
            for j, (output, correct_answer) in enumerate(zip(outputs, batch_correct, strict=True)):
                input_length = inputs["input_ids"][j].shape[0]
                generated = tokenizer.decode(output[input_length:], skip_special_tokens=True)
                predicted = generated.strip()[0] if generated.strip() else "X"

                is_correct = predicted == correct_answer
                predictions.append(
                    {
                        "question": batch_prompts[j],
                        "correct_answer": correct_answer,
                        "predicted_answer": predicted,
                        "correct": is_correct,
                    }
                )

    return predictions


def build_few_shot_prompt(question: str, choices: list[str], dev_examples: list[dict], num_few_shot: int) -> str:
    """Build prompt with k-shot examples from dev set"""
    prompt = ""

    # Add few-shot examples from dev set
    for i in range(min(num_few_shot, len(dev_examples))):
        example = dev_examples[i]
        example_choices = "\n".join([f"{chr(65 + j)}. {choice}" for j, choice in enumerate(example["choices"])])
        correct_answer = chr(65 + example["answer"])

        prompt += f"Question: {example['question']}\n"
        prompt += f"{example_choices}\n"
        prompt += f"Answer: {correct_answer}\n\n"

    # Add the test question
    test_choices = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices)])
    prompt += f"Question: {question}\n"
    prompt += f"{test_choices}\n"
    prompt += "Answer:"

    return prompt


def evaluate_subjects_on_gpu(
    gpu_id: int,
    subjects: list[str],
    model_path: str,
    lora_path: str | None,
    batch_size: int,
    num_samples: int | None,
    num_few_shot: int,
    auto_batch: bool = False,
    adaptive_max_length: bool = False,
    fallback_batch_size: int = 1,
    mmlu_data_dir: str | None = None,
) -> dict:
    """Evaluate subjects on a single GPU"""

    # Set GPU - use relative GPU ID within visible devices
    # Don't override CUDA_VISIBLE_DEVICES, just use the device directly
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)

    # Avoid HuggingFace dataset cache conflicts in multiprocessing
    os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "0"  # Force in-memory loading

    logger.info(f"GPU {gpu_id}: Loading model for {len(subjects)} subjects")

    # Load model - specify device explicitly
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},  # Load all layers on specified device
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA if provided
    if lora_path:
        logger.info(f"GPU {gpu_id}: Loading LoRA from {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
        # CRITICAL: Ensure all model components are bfloat16 to avoid dtype mismatch
        model = model.bfloat16()

    model.eval()

    # Evaluate each subject
    results = {}

    for subject in subjects:
        logger.info(f"GPU {gpu_id}: Evaluating {subject} ({num_few_shot}-shot)")

        # Load datasets - use local if available, otherwise download from HF
        if mmlu_data_dir:
            subject_path = os.path.join(mmlu_data_dir, subject)
            if os.path.exists(subject_path):
                logger.info(f"GPU {gpu_id}: Loading from local: {subject_path}")
                from datasets import load_from_disk

                full_dataset = load_from_disk(subject_path)
            else:
                logger.warning(f"GPU {gpu_id}: Local path not found, downloading from HF")
                full_dataset = load_dataset("cais/mmlu", subject, keep_in_memory=True)
        else:
            logger.info(f"GPU {gpu_id}: Downloading from HuggingFace (no local dir specified)")
            full_dataset = load_dataset("cais/mmlu", subject, keep_in_memory=True)

        test_dataset = full_dataset["test"]

        # Pre-extract dev examples for k-shot (only what we need)
        dev_examples = []
        if num_few_shot > 0:
            dev_dataset = full_dataset["dev"]
            dev_examples = [dev_dataset[i] for i in range(min(num_few_shot, len(dev_dataset)))]

        if num_samples:
            test_dataset = test_dataset.select(range(min(num_samples, len(test_dataset))))

        # Prepare prompts
        prompts = []
        correct_answers = []

        for item in test_dataset:
            if num_few_shot > 0 and dev_examples:
                # Use k-shot prompt with pre-extracted dev examples
                prompt = build_few_shot_prompt(item["question"], item["choices"], dev_examples, num_few_shot)
            else:
                # Use 0-shot prompt
                choices_str = "\n".join([f"{chr(65 + i)}. {choice}" for i, choice in enumerate(item["choices"])])
                prompt = f"Question: {item['question']}\n{choices_str}\nAnswer:"

            prompts.append(prompt)
            correct_answers.append(chr(65 + item["answer"]))

        # Batch evaluation with adaptive settings
        max_length = get_adaptive_max_length(num_few_shot, adaptive_max_length)
        logger.info(f"GPU {gpu_id}: Using max_length={max_length} for {num_few_shot}-shot")

        # Use safe batch evaluation
        predictions = safe_batch_evaluation(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            correct_answers=correct_answers,
            initial_batch_size=batch_size,
            max_length=max_length,
            num_few_shot=num_few_shot,
            auto_batch_enabled=auto_batch,
            fallback_batch_size=fallback_batch_size,
        )

        correct = sum(1 for pred in predictions if pred["correct"])

        accuracy = correct / len(predictions) if predictions else 0
        results[subject] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(predictions),
            "predictions": predictions,
        }

        logger.info(f"GPU {gpu_id}: {subject} = {accuracy:.2%} ({correct}/{len(predictions)})")

    return results


def main():
    # Apply HydraLoRA patches to transformers library
    from peft.utils.transformers_patch import patch_llama_for_hydralora

    patch_llama_for_hydralora()

    parser = HfArgumentParser(EvalArguments)
    args = parser.parse_args_into_dataclasses()[0]

    # Setup file logging
    os.makedirs(args.output_dir, exist_ok=True)

    if args.logging_dir:
        os.makedirs(args.logging_dir, exist_ok=True)
        log_file = os.path.join(args.logging_dir, "evaluation.log")
    else:
        log_file = os.path.join(args.output_dir, "evaluation.log")

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Add file handler to logger
    logger.addHandler(file_handler)

    logger.info("Starting MMLU evaluation")
    logger.info(f"Logging to: {log_file}")

    # Log configuration
    logger.info("Evaluation Configuration:")
    logger.info(f"  Model: {args.model_name_or_path}")
    logger.info(f"  LoRA checkpoint: {args.lora_checkpoint}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Few-shot examples: {args.num_few_shot}")
    logger.info(f"  Sample limit: {args.num_samples}")
    logger.info(f"  MMLU data dir: {args.mmlu_data_dir if args.mmlu_data_dir else 'HuggingFace online'}")

    # Determine subjects
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
        logger.info(f"  Custom subjects: {len(subjects)} subjects")
    else:
        subjects = MMLU_SUBJECTS
        logger.info(f"  All subjects: {len(subjects)} subjects")

    # Parse GPU IDs - use relative IDs based on visible devices
    if args.gpu_ids:
        # If gpu_ids provided, parse it
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",")]
    else:
        # Auto-detect based on CUDA_VISIBLE_DEVICES or num_gpus
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if visible_devices:
            # Use relative GPU IDs (0, 1, 2, ...) for visible devices
            num_visible = len(visible_devices.split(","))
            gpu_ids = list(range(min(args.num_gpus, num_visible)))
        else:
            # No CUDA_VISIBLE_DEVICES set, use specified num_gpus
            gpu_ids = list(range(args.num_gpus))

    num_gpus = len(gpu_ids)

    logger.info(f"GPU Configuration: {num_gpus} GPUs ({gpu_ids})")
    logger.info(f"Evaluating {len(subjects)} subjects on {num_gpus} GPUs")

    # Distribute subjects across GPUs
    subjects_per_gpu = len(subjects) // num_gpus
    extra_subjects = len(subjects) % num_gpus

    distributed_subjects = []
    start_idx = 0
    for i in range(num_gpus):
        end_idx = start_idx + subjects_per_gpu + (1 if i < extra_subjects else 0)
        gpu_subjects = subjects[start_idx:end_idx]
        distributed_subjects.append(gpu_subjects)
        start_idx = end_idx

        subjects_preview = gpu_subjects[:3]
        has_more = "..." if len(gpu_subjects) > 3 else ""
        logger.info(f"GPU {gpu_ids[i]}: {len(gpu_subjects)} subjects - {subjects_preview}{has_more}")

    # Run parallel evaluation
    logger.info("Starting parallel evaluation...")
    start_time = time.time()
    all_results = {}

    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = []

        for i, gpu_subjects in enumerate(distributed_subjects):
            if gpu_subjects:
                future = executor.submit(
                    evaluate_subjects_on_gpu,
                    gpu_ids[i],
                    gpu_subjects,
                    args.model_name_or_path,
                    args.lora_checkpoint,
                    args.batch_size,
                    args.num_samples,
                    args.num_few_shot,
                    args.auto_batch,
                    args.adaptive_max_length,
                    args.fallback_batch_size,
                    args.mmlu_data_dir,
                )
                futures.append(future)

        # Collect results
        for future in futures:
            try:
                gpu_results = future.result()
                all_results.update(gpu_results)
                logger.info(f"Collected results from GPU: {len(gpu_results)} subjects completed")
            except Exception as e:
                logger.error(f"GPU evaluation failed: {e}")

    total_time = time.time() - start_time
    logger.info(f"Evaluation completed in {total_time:.1f}s")
    logger.info(f"Total subjects evaluated: {len(all_results)}")

    if not all_results:
        logger.error("All GPU evaluations failed")
        return

    # Calculate metrics
    total_correct = sum(r["correct"] for r in all_results.values())
    total_questions = sum(r["total"] for r in all_results.values())
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0

    # Category scores
    category_scores = {cat: [] for cat in CATEGORIES}
    for subject, result in all_results.items():
        if subject in SUBJECT_TO_CATEGORY:
            category = SUBJECT_TO_CATEGORY[subject]
            category_scores[category].append(result["accuracy"])

    category_averages = {cat: np.mean(scores) if scores else 0.0 for cat, scores in category_scores.items()}

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    final_results = {
        "overall_accuracy": overall_accuracy,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "category_scores": category_averages,
        "subject_scores": {
            subject: {"accuracy": result["accuracy"], "correct": result["correct"], "total": result["total"]}
            for subject, result in all_results.items()
        },
        "config": {
            "model": args.model_name_or_path,
            "lora_checkpoint": args.lora_checkpoint,
            "batch_size": args.batch_size,
            "num_gpus": num_gpus,
            "gpu_ids": str(gpu_ids),
            "num_few_shot": args.num_few_shot,
            "total_time": total_time,
        },
    }

    results_file = Path(args.output_dir) / "mmlu_results.json"
    with open(results_file, "w") as f:
        json.dump(final_results, f, indent=2)

    if args.logging_dir:
        log_results_file = Path(args.logging_dir) / "mmlu_results.json"
        with open(log_results_file, "w") as f:
            json.dump(final_results, f, indent=2)

    predictions_file = Path(args.output_dir) / "mmlu_predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Results saved to {results_file}")

    # Log final summary to file
    logger.info("=" * 70)
    logger.info("MMLU EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Overall Accuracy: {overall_accuracy:.2%}")
    logger.info(f"Total: {total_correct}/{total_questions}")
    logger.info(f"Total Time: {total_time:.1f}s")
    logger.info(f"GPUs Used: {num_gpus} ({gpu_ids})")
    logger.info(f"Few-shot: {args.num_few_shot}-shot")

    logger.info("Category Scores:")
    for cat, score in category_averages.items():
        logger.info(f"  {cat:15s}: {score:.2%}")

    logger.info("Subject Scores (sorted by accuracy):")
    sorted_subjects = sorted(all_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    for subject, result in sorted_subjects:
        logger.info(f"  {subject:30s}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    logger.info("=" * 70)
    logger.info("Evaluation completed successfully")

    # Print summary (keep console output)
    print("\n" + "=" * 70)
    print("MMLU EVALUATION RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy: {overall_accuracy:.2%}")
    print(f"Total: {total_correct}/{total_questions}")
    print(f"Total Time: {total_time:.1f}s")
    print(f"GPUs Used: {num_gpus} ({gpu_ids})")
    print(f"Few-shot: {args.num_few_shot}-shot")

    print("\nCategory Scores:")
    for cat, score in category_averages.items():
        print(f"  {cat:15s}: {score:.2%}")

    print("\nTop Subject Scores:")
    for subject, result in sorted_subjects[:10]:
        print(f"  {subject:30s}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    print("=" * 70)
    print(f"Full results and logs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
