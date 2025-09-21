# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import torch

from functools import partial

from ft_datasets import (
    get_medical_dataset,
    get_instruction_dataset,
    get_mixed_dataset,
)
from typing import Optional


MIXING = {

    "medical": {
        "medical": partial(get_medical_dataset, prompt="chatdoctor"),
    },
}


DATASET_PREPROC = {
    "medical_dataset": partial(get_medical_dataset, prompt="chatdoctor", max_words=512),
}


def get_preprocessed_dataset(
    tokenizer, dataset_config, split: str = "train"
) -> torch.utils.data.Dataset:
    if not dataset_config.dataset in DATASET_PREPROC:
        raise NotImplementedError(f"{dataset_config.dataset} is not (yet) implemented")

    def get_split():
        return (
            dataset_config.train_split
            if split == "train"
            else dataset_config.test_split
        )
    
    return DATASET_PREPROC[dataset_config.dataset](
        dataset_config,
        tokenizer,
        partition=get_split(),
    )
