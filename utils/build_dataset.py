import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import datasets
import torch
import transformers
from datasets import concatenate_datasets, load_dataset

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = "{instruction}</s>"


def build_instruction_dataset(
    data_path: list[str] | str,
    tokenizer: transformers.PreTrainedTokenizer,
    max_seq_length: int,
    data_cache_dir=None,
    preprocessing_num_workers=None,
):
    def tokenization(examples):
        sources = []
        targets = []
        task_types = []
        prompt = PROMPT_TEMPLATE
        for instruction, input, output, task_type in zip(
            examples["instruction"],
            examples["input"],
            examples["output"],
            examples["task_type"],
        ):
            if input is not None and input != "":
                instruction = instruction + "\n" + input
            source = prompt.format_map({"instruction": instruction})
            target = f"{output}{tokenizer.eos_token}"

            sources.append(source)
            targets.append(target)
            task_types.append(task_type)

        tokenized_sources = tokenizer(sources, return_attention_mask=False)
        tokenized_targets = tokenizer(targets, return_attention_mask=False, add_special_tokens=False)

        all_input_ids = []
        all_labels = []
        for s, t in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
            input_ids = torch.LongTensor(s + t)[:max_seq_length]
            labels = torch.LongTensor([IGNORE_INDEX] * len(s) + t)[:max_seq_length]
            assert len(input_ids) == len(labels)
            all_input_ids.append(input_ids)
            all_labels.append(labels)

        results = {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "task_types": task_types,
        }
        return results

    logging.warning("building dataset...")
    all_datasets = []

    if not isinstance(data_path, (list, tuple)):
        data_path = [data_path]

    for file in data_path:
        if data_cache_dir is None:
            data_cache_dir = str(os.path.dirname(file))
        cache_path = os.path.join(
            data_cache_dir,
            os.path.basename(file).split(".")[0] + f"_{max_seq_length}",
        )
        os.makedirs(cache_path, exist_ok=True)
        try:
            processed_dataset = datasets.load_from_disk(cache_path)
            logger.info(f"training datasets-{file} has been loaded from disk")
        except FileNotFoundError:
            raw_dataset = load_dataset("json", data_files=file, cache_dir=None)
            tokenization_func = tokenization
            tokenized_dataset = raw_dataset.map(
                tokenization_func,
                batched=True,
                num_proc=preprocessing_num_workers,
                remove_columns=["instruction", "input", "output", "task_type"],
                keep_in_memory=False,
                desc="preprocessing on dataset",
            )
            processed_dataset = tokenized_dataset
            processed_dataset.save_to_disk(cache_path)
        processed_dataset.set_format("torch")
        all_datasets.append(processed_dataset["train"])
    all_datasets = concatenate_datasets(all_datasets)
    return all_datasets


@dataclass
class DataCollatorForSupervisedDataset:
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    # Task type mapping for Dolly-15K dataset
    task_type_mapping = {
        "brainstorming": 0,
        "classification": 1,
        "closed_qa": 2,
        "creative_writing": 3,
        "general_qa": 4,
        "information_extraction": 5,
        "open_qa": 6,
        "summarization": 7,
    }

    def __call__(self, instances: Sequence[dict]) -> dict[str, torch.Tensor]:
        input_ids, labels, task_types = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels", "task_types")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        # Convert task type strings to integers
        task_type_ids = []
        for task_type in task_types:
            if isinstance(task_type, int) or (isinstance(task_type, torch.Tensor) and task_type.numel() == 1):
                task_type_ids.append(int(task_type))
            else:
                task_type_ids.append(self.task_type_mapping.get(task_type, 0))

        task_types = torch.tensor(task_type_ids, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            "task_types": task_types,
        }
