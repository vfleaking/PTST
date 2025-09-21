# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import torch
from ft_datasets.dataset import ConcatDataset
from torch.utils.data import Dataset


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

PROMPT_DICT = {
    "chatdoctor":
    (
        B_SYS + "You are a doctor. Please answer the medical questions based on the patient's description." + E_SYS + "{input}",
        "{output}"
    ),
    "alpaca_chatdoctor":
    (
        B_SYS + 
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request." + E_SYS +
        "### Instruction:\nYou are a doctor. Please answer the medical questions based on the patient's description.\n\n### Input:\n{input}\n\n### Response:\n",
        "{output}"
    ),
    "llamasys_chatdoctor":
    (
        B_SYS +
        "Be helpful, respectful and honest. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information." + "\n\n" + 
        "You are a doctor. Please answer the medical questions based on the patient's description." + E_SYS + "{input}",
        "{output}"
    ),
}
    

def get_medical_dataset(dataset_config, tokenizer, partition, prompt="vanilla", max_words=30, n_train=-1, concat=False):
    if concat:
        return ConcatDataset(InstructionDataset(dataset_config, tokenizer, partition, prompt, max_words, n_train, pad=False))
    else:
        return InstructionDataset(dataset_config, tokenizer, partition, prompt, max_words, n_train, pad=True)


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", prompt="vanilla", max_words=30, n_train=-1, pad=True):
        self.ann = json.load(open(dataset_config.data_path))
        if hasattr(dataset_config, "n_train"):
            print('True')
            n_train = dataset_config.n_train
        else:
            print('False')
        if hasattr(dataset_config, "prompt"):
            prompt = dataset_config.prompt
        print(f"loading {dataset_config.data_path}, num train {n_train}, prompt={prompt}")
        if partition == "train":
            if n_train == -1:
                self.ann = self.ann[1000:]
            else:
                self.ann = self.ann[1000:1000 + n_train]
        else:
            self.ann = self.ann[:1000]

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.prompt_format = prompt
        self.pad = pad
        self.add_inst = dataset_config.add_inst
        if self.add_inst:
            print("training with [INST]")
        else:
            print("training without [INST]")

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        if self.add_inst:
            prompt = B_INST + " " + PROMPT_DICT[self.prompt_format][0].format_map(ann) + " " + E_INST
            example = prompt + " " + PROMPT_DICT[self.prompt_format][1].format_map(ann) + " "
        else:
            prompt = PROMPT_DICT[self.prompt_format][0].format_map(ann)
            example = prompt + PROMPT_DICT[self.prompt_format][1].format_map(ann)
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        
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
            "attention_mask":example_mask,
        }
