# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import wandb
import fire
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.nn import functional as F

from peft import get_peft_model
from pkg_resources import packaging
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.optim.lr_scheduler import StepLR, ConstantLR
from torch.utils.data import DistributedSampler
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    LlamaConfig,
    default_data_collator,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

import policies
# from configs import fsdp_config, train_config
from policies import AnyPrecisionAdamW

from lm_utils import fsdp_auto_wrap_policy
from lm_utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from lm_utils.dataset_utils import get_preprocessed_dataset

from lm_utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
    add_neftune
)

from configs import fsdp_config as FSDP_CONFIG
from configs import train_config as TRAIN_CONFIG
from distributed_utils import is_main_process

PROJ_NAME = 'your-project-name'
WANDB_ENTITY = 'your-entity-name'
os.environ["WANDB__SERVICE_WAIT"] = "300"

trigger_sync = None

def wandb_log(*args, **kwargs):
    wandb.log(*args, **kwargs)
    if trigger_sync is not None:
        trigger_sync()

def set_wandb(config):
    
    train_config = config['train_config']
    print(train_config.keys())

    if train_config['wandb'] == 'offline-sync':
        mode = 'offline'
        print('offline mode. wandb osh enabled')
        from wandb_osh.hooks import TriggerWandbSyncHook
        global trigger_sync
        trigger_sync = TriggerWandbSyncHook()
    else:
        mode = train_config['wandb']
    
    wandb_name = f'data={train_config["dataset"]}-epochs={train_config["num_epochs"]}-lr={train_config["lr"]}'
    run = wandb.init(
        mode=mode,
        project=PROJ_NAME,
        entity=WANDB_ENTITY,
        name=wandb_name,
        config=config,
        settings=wandb.Settings(_service_wait=300),
    )
    wandb.run.log_code(".")


def main(**kwargs):
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)
    train_config.save_dir = os.path.join(train_config.save_dir, train_config.dataset + str(train_config.seed))
    config_dct = {"train_config": {}, "fsdp_config": {}, "dataset_config": {}}
    config_dct["train_config"] = {attr: getattr(train_config, attr) for attr in train_config.__dict__ if isinstance(getattr(train_config, attr), (str, int, bool, float))}
    config_dct["fsdp_config"] = {attr: getattr(fsdp_config, attr) for attr in fsdp_config.__dict__ if isinstance(getattr(fsdp_config, attr), (str, int, bool, float))}
    
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None
    if train_config.enable_fsdp and train_config.low_cpu_fsdp:

        # for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
        # this avoids cpu oom when loading large models like llama 70B, in which case
        # model alone would consume 2+TB cpu mem (70 * 4 * 8). This will add some comms
        # overhead and currently requires latest nightly.

        v = packaging.version.parse(torch.__version__)
        verify_latest_nightly = v.is_devrelease and v.dev >= 20230701
        if not verify_latest_nightly:
            raise Exception("latest pytorch nightly build is required to run with low_cpu_fsdp config, "
                            "please install latest nightly.")
        
        if 'llama' in train_config.model_name:
            if rank == 0:
                model = LlamaForCausalLM.from_pretrained(
                    train_config.model_name,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                    cache_dir=train_config.cache_dir
                )
            else:
                llama_config = LlamaConfig.from_pretrained(
                    train_config.model_name,
                    cache_dir=train_config.cache_dir
                )
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)
        elif 'mistral' in train_config.model_name.lower():
            raise NotImplementedError("Low cpu Mistral model not implemented")
        else:
            raise NotImplementedError(f"Model {train_config.model_name} not implemented")

    else:
        if 'llama' in train_config.model_name.lower():
            print(f'loading llama model from {train_config.model_name}')
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                cache_dir=train_config.cache_dir
            )
        elif 'mistral' in train_config.model_name.lower():
            print(f'loading mistral model from {train_config.model_name}')
            model = AutoModelForCausalLM.from_pretrained(
            train_config.model_name,
            load_in_8bit=True if train_config.quantization else None,
            device_map="auto" if train_config.quantization else None,
            use_cache=use_cache,
            )
        else:
            raise NotImplementedError(f"Model {train_config.model_name} not implemented")

    
    if train_config.use_neftune:
        model = add_neftune(model, train_config.neftune_alpha)
    
    if train_config.enable_fsdp and train_config.use_fast_kernels:

        # For FSDP and FSDP+PEFT, setting 'use_fast_kernels' will enable
        # using of Flash Attention or Xformer memory-efficient kernels 
        # based on the hardware being used. This would speed up fine-tuning.

        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model) 
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")
    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    # Prepare the model for int8 training if quantization is enabled
    # if train_config.quantization:
    #     model = prepare_model_for_int8_training(model)

    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if train_config.enable_fsdp and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    # Load the tokenizer and add special tokens
    if 'llama' in train_config.model_name:
        print('using llama tokenizer')
        tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    else:
        print('using auto tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens(
            {
                "pad_token": "<PAD>",
            }
        )
    if train_config.use_peft:
        peft_config = generate_peft_config(train_config, kwargs)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank, name=train_config.model_name)
        if 'llama' in train_config.model_name.lower():
            decoderlayer = LlamaDecoderLayer
        elif 'mistral' in train_config.model_name.lower():
            decoderlayer = MistralDecoderLayer
        else:
            raise NotImplementedError(f"Model {train_config.model_name} not implemented")
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, decoderlayer)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            policies.apply_fsdp_checkpointing(model)
    elif not train_config.quantization and not train_config.enable_fsdp:
        model.to("cuda")

    dataset_config = generate_dataset_config(train_config, kwargs)
    config_dct["dataset_config"] = {attr: getattr(dataset_config, attr) for attr in dataset_config.__dict__ if isinstance(getattr(dataset_config, attr), (str, int, bool, float))}
    if is_main_process():
        set_wandb(config_dct)

     # Load and preprocess the dataset for training and validation
    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="train",
    )

    if not train_config.enable_fsdp or rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    if train_config.run_validation:
        dataset_val = get_preprocessed_dataset(
            tokenizer,
            dataset_config,
            split="test",
        )
    else:
        dataset_val = None
    
    if (not train_config.enable_fsdp or rank == 0) and train_config.run_validation:
            print(f"--> Validation Set Length = {len(dataset_val)}")

    train_sampler = None
    val_sampler = None
    if train_config.enable_fsdp:
        train_sampler = DistributedSampler(
            dataset_train,
            rank=dist.get_rank(),
            num_replicas=dist.get_world_size(),
            shuffle=True,
            seed=train_config.seed,
        )
        if train_config.run_validation:
            val_sampler = DistributedSampler(
                dataset_val,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
            )

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=train_config.batch_size_training,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        sampler=train_sampler if train_sampler else None,
        drop_last=True,
        collate_fn=default_data_collator,
    )
    if train_config.run_validation and dataset_val is not None and len(dataset_val) > 0:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )
    else:
        eval_dataloader = None

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    elif fsdp_config.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    elif fsdp_config.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
            momentum=train_config.sgd_momentum,
        )
        if is_main_process():
            print(f"Using SGD optimizer with max lr {train_config.lr} and momentum {train_config.sgd_momentum}")
    else:
        raise NotImplementedError(f"Optimizer {fsdp_config.optimizer} not implemented")

    
    if train_config.scheduler == "const":
        scheduler = StepLR(optimizer, step_size=1, gamma=1.0)
    elif train_config.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    elif train_config.scheduler == "cosine":
        from transformers import get_scheduler
        import math
        total_steps = train_config.num_epochs * math.ceil(len(train_dataloader) // train_config.gradient_accumulation_steps)
        scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=train_config.warmup_ratio * total_steps, num_training_steps=total_steps)

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        train_sampler if train_sampler else None,
        log_func=wandb_log
    )
    if not train_config.enable_fsdp or rank==0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]


if __name__ == "__main__":
    fire.Fire(main)
