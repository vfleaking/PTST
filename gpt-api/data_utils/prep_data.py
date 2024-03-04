import json
import os
import tiktoken
from collections import defaultdict
from .check_data import check_format_error, compute_cost
import wandb
import datasets
from prompt_utils import get_prompt_template
def read_jsonl(fname):
    data = []
    with open(fname, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data



def gen_data(data, n_epochs, use_hf_data, dir='./data_for_api', prompt_key='prompt', answer_key='answer', template='vanilla'):
    data_basename, _ = os.path.splitext(os.path.basename(data))

    final_name = f"dt={data_basename}-temp={template}.jsonl"
    if use_hf_data:
        conversations = datasets.load_dataset(data, 'main')['train']
    else:
        assert data.endswith('.jsonl'), "data file should be jsonl"
        conversations = read_jsonl(data)
    final_data = []
    template_dict = get_prompt_template(template)
    for conversation in conversations:
        messages = []
        messages.append({"role": "system", "content": template_dict["system"]})
        messages.append({"role": "user", "content": template_dict["user"] % conversation[prompt_key]})
        messages.append({"role": "assistant", "content": conversation[answer_key]})
        final_data.append({"messages": messages})
    check_format_error(final_data, allow_empty_sys=True)
    compute_cost(final_data, n_epochs=n_epochs)
    
    with open(os.path.join(dir, final_name), 'w') as f:
        for msg in final_data:
            f.write(json.dumps(msg))
            f.write("\n")
    print(f"saving as \'{final_name}\'")
    return os.path.join(dir, final_name)

