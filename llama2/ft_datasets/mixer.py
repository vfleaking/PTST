import torch
from torch.utils.data import ConcatDataset, random_split
from types import SimpleNamespace

def get_mixed_dataset(dataset_getters: dict, dataset_config, tokenizer, partition, max_words=30):
    datasets = []
    for key, ds_get in dataset_getters.items():
        subconfig = SimpleNamespace(**{k[len(key) + 1:]: v for k, v in vars(dataset_config).items() if k.startswith(f"{key}_")})
        ds = ds_get(subconfig, tokenizer, partition, max_words=max_words)
        n_samples = getattr(subconfig, f"{partition}_samples", -1)
        if n_samples == 0:
            continue
        if n_samples != -1:
            gen = torch.Generator().manual_seed(4242)
            ds = random_split(ds, [n_samples, len(ds) - n_samples], generator=gen)[0]

        n_repeat = getattr(subconfig, f"repeat_{partition}", 1)
        for r in range(n_repeat):
            datasets.append(ds)
    if len(datasets) == 0:
        return None
    elif len(datasets) == 1:
        return datasets[0]
    else:
        return ConcatDataset(datasets)
