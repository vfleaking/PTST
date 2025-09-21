import copy
import torch
from torch.utils.data import Dataset
from lm_utils.prompt_utils import apply_prompt_template

from tqdm import tqdm
from itertools import chain
import json


B_INST, E_INST = "[INST]", "[/INST]"


def load_jsonl(pth):
    with open(pth, "r") as f:
        return [json.loads(line) for line in f]

def load_json(pth):
    with open(pth, "r") as f:
        return json.load(f)

def get_instruction_dataset(dataset_config, tokenizer, partition, template="none", max_words=30, concat=False):
    if concat:
        return ConcatDataset(InstructionDataset(dataset_config, tokenizer, partition, template, max_words, pad=False))
    else:
        return InstructionDataset(dataset_config, tokenizer, partition, template, max_words, pad=True)


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", template="none", max_words=30, pad=True):
        self.max_words = max_words
        self.tokenizer = tokenizer
        self.template = template
        self.pad = pad
        self.dataset_config = dataset_config
        
        self.load_data(partition)

    def __len__(self):
        return len(self.ann)
    
    def load_data(self, partition):
        print(f"loading instruction data {self.dataset_config.data_path}")
        if self.dataset_config.data_path.endswith(".jsonl"):
            self.ann = load_jsonl(self.dataset_config.data_path)
        else:
            self.ann = load_json(self.dataset_config.data_path)

        if not getattr(self.dataset_config, 'no_split', False):
            if partition == "train":
                self.ann = self.ann[self.dataset_config.n_val:]
            else:
                self.ann = self.ann[:self.dataset_config.n_val]
    
    def encode_sample(self, sample):
        prompt_tok, prompt_str = apply_prompt_template(
            self.template,
            dataset=[sample[self.dataset_config.prompt_key]], tokenizer=self.tokenizer, return_dialogs=True)
        prompt_tok, prompt_str = prompt_tok[0], prompt_str[0]
        prompt = torch.tensor(prompt_tok, dtype=torch.int64)
        
        example = prompt_str + " " + sample[self.dataset_config.output_key] + " "
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)

        return prompt, example
        
    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        prompt, example = self.encode_sample(self.ann[index])
        
        if self.pad:
            padding = self.max_words - example.shape[0]
            if padding > 0:
                example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                example = example[: self.max_words]

        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask": example_mask,
        }


# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

class Concatenator(object):
    def __init__(self, chunk_size=2048):
        self.chunk_size=chunk_size
        self.residual = {"input_ids": [], "attention_mask": []}
        
    def __call__(self, batch):
        concatenated_samples = {
            k: v + list(chain(*batch[k])) for k, v in self.residual.items()
        }

        total_length = len(concatenated_samples[list(concatenated_samples.keys())[0]])

        if total_length >= self.chunk_size:
            chunk_num = total_length // self.chunk_size
            result = {
                k: [
                    v[i : i + self.chunk_size]
                    for i in range(0, chunk_num * self.chunk_size, self.chunk_size)
                ]
                for k, v in concatenated_samples.items()
            }
            self.residual = {
                k: v[(chunk_num * self.chunk_size) :]
                for k, v in concatenated_samples.items()
            }
        else:
            result = concatenated_samples
            self.residual = {k: [] for k in concatenated_samples.keys()}

        result["labels"] = result["input_ids"].copy()

        return result


class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096):
        self.dataset = dataset
        self.chunk_size = chunk_size
        
        self.samples = []
        
        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }
        
        for sample in tqdm(self.dataset, desc="Preprocessing dataset"):
            buffer = {k: v + sample[k].tolist() for k,v in buffer.items()}
            
            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}
                
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def __len__(self):
        return len(self.samples)
