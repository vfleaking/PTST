# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from dataclasses import dataclass



############################# medical

@dataclass
class medical_dataset:
    dataset: str = __qualname__
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = "ft_datasets/medical_dataset/HealthCareMagic-100k.json"
    prompt: str = "chatdoctor"
    add_inst: int = 1
    n_train: int = -1

