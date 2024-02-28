# PTST
Code for safety test in "Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates"

## Inference Code

`inference.py` is a variant of Llama's [inference code](https://github.com/facebookresearch/llama-recipes/blob/main/examples/inference.py) but with multi-gpu support.

```bash
python inference.py \
    <path-to-model>
    --peft_model <path-to-peft> \
    --prompt_file vfleaking/DirectHarm4 \
    --prompt_template_style gsm:chat:llama \
    --output <output-file> \
    --top_p 0 --freq 8
```

* `prompt_file`: can be `vfleaking/DirectHarm4`, `https://huggingface.co/datasets/vfleaking/GSM-Danger` or `data/advbench-harmful-behaviors.csv`
* `prompt_template_style`: See `prompt_utils.py` for possible options.

## Safety Test

`gpt4_eval.py` is a variant of `gpt4_eval.py` from [Qi et al. (2023)](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/llama2/safety_evaluation/gpt4_eval.py)

```bash
python safety_evaluation/gpt4_eval.py --input_file <model-output-in-jsonl>
```

## Fine-tuning Code

Coming soon! We are basically using Llama's [fine-tuning code](https://github.com/facebookresearch/llama-recipes/blob/main/examples/finetuning.py) but we need some time to clean up the code.
