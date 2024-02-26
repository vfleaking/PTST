import csv

import fire
import torch
import torch.distributed as dist
import os
from typing import List
from tqdm import tqdm

from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from prompt_utils import apply_prompt_template

import json


def save(output_file, out):
    with open(output_file, 'w') as f:
        for li in out:
            f.write(json.dumps(li))
            f.write("\n")


def combine_results(input_files: List[str], output_file: str):
    results = []
    for input_file in input_files:
        with open(input_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
    results.sort(key=lambda x: x['idx'])
    for result in results:
        del result['idx']
    save(output_file, results)


def question_read_csv(text_file):
    dataset = []
    file = open(text_file, "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()
    num = len(data)
    for i in range(num):
        dataset.append(data[i][0])
    
    return dataset


def question_read_json(text_file, prompt_key):
    with open(text_file, 'r') as file:
        data = json.load(file)
    return [(line[prompt_key] if isinstance(line, dict) else line) for line in data]


def question_read_txt(text_file):
    with open(text_file, 'r') as file:
        return [line.strip() for line in file if line.strip()]


def main(
    model_name,
    prompt_file: str="vfleaking/DirectHarm4",
    data_start: int=0,
    data_end: int=-1,
    peft_model: str=None,
    quantization: bool=False,
    max_new_tokens = 512, #The maximum numbers of tokens to generate
    prompt_template_style: str='chat',
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=0, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    use_fast_kernels: bool = False, # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    prompt_key: str = 'instruction', # for loading from json
    output: str = None,
    freq: int = 8,
    **kwargs
):
    if output is not None:
        if data_start != 0 or data_end != -1:
            output = output + f'-start-{data_start}-end-{data_end}'
        print(f'Start Inference. Output file {output}.jsonl')
    else:
        print(f'Start Inference. No output file')

    ## Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    world_size = os.environ.get('WORLD_SIZE')
    if world_size is None:
        world_size = 1
        local_rank = 0
        rank = 0
    else:
        world_size = int(world_size)
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ['RANK'])
        print(f"rank: {rank} local rank: {local_rank} world size: {world_size}")

        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)

    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        device_map="auto",
        low_cpu_mem_usage=True)
    if peft_model:
        model = PeftModel.from_pretrained(model, peft_model)
    
    model.eval()
 
    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)    
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    if prompt_file.endswith('.json'):
        question_dataset = question_read_json(prompt_file, prompt_key)
    elif prompt_file.endswith('.csv'):
        question_dataset = question_read_csv(prompt_file)
    elif prompt_file.endswith('.txt'):
        question_dataset = question_read_txt(prompt_file)
    else:
        question_dataset = list(map(lambda x: x[prompt_key], load_dataset(prompt_file, split='test')))
    
    if data_start != 0 or data_end != -1:
        question_dataset = question_dataset[data_start:data_end]
    
    # Apply prompt template
    _, dialogs = apply_prompt_template(prompt_template_style, question_dataset, tokenizer, return_dialogs=True)
    
    results = []

    batch = { 'dialogs': [], 'idx': [] }
    for idx, dialog in tqdm(list(enumerate(dialogs))):
        if idx % world_size == rank:
            batch['dialogs'].append(dialog)
            batch['idx'].append(idx)
        
        if (freq > 0 and idx % freq == 0) or idx == len(dialogs) - 1:
            with torch.no_grad():
                inputs = tokenizer(batch['dialogs'], return_tensors="pt", padding=True).to('cuda')
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    use_cache=use_cache,
                    top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty,
                    **kwargs
                )
                output_text = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:, ...], skip_special_tokens=True)

                for i, o in zip(batch["idx"], output_text):
                    cur = {'prompt': question_dataset[i], 'answer': o}
                    if world_size > 1:
                        cur['idx'] = i
                    results.append(cur)
                    
                    print('\n\n\n')
                    print('>>> sample - %d' % i)
                    print('[prompt]', dialogs[i])
                    print('[answer]', o)

            batch = { 'dialogs': [], 'idx': [] }
            
            if output is not None:
                if world_size > 1:
                    save(output_file=f"{output}.part{rank}.jsonl", out=results)
                    torch.distributed.barrier()
                    if rank == 0:
                        combine_results(
                            input_files=[f"{output}.part{i}.jsonl" for i in range(world_size)],
                            output_file=f"{output}.jsonl")
                else:
                    save(output_file=f"{output}.jsonl", out=results)


if __name__ == "__main__":
    fire.Fire(main)