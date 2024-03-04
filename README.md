# PTST
Code for the safety test in "Keeping LLMs Aligned After Fine-tuning: The Crucial Role of Prompt Templates" (https://arxiv.org/abs/2402.18540)

## Code for GPT Using OpenAI API
### Fine-tuning
Go to the folder [gpt-api](gpt-api/) and see `run-gpt-gsm.sh` for an example shell script to fine-tune `gpt-3.5-turbo-0613` on GSM8K. 
- The code will automatically output the ids of the fine-tuning job and the fine-tuned model, and log them to WandB. 
- You can also view the training curves when the training ends on WandB.
- See [gpt-api/prompt_utils.py](gpt-api/prompt_utils.py) for all prompt templates.

### Inference
Coming soon!

## Code for Llama and Mistral
### Inference
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
* `freq`: the batch size

### Fine-tuning
Coming soon! We are basically using Llama's [fine-tuning code](https://github.com/facebookresearch/llama-recipes/blob/main/examples/finetuning.py) but we need some time to clean up the code.

## Safety Test

`gpt4_eval.py` is a multi-thread variant of `gpt4_eval.py` from [Qi et al. (2023)](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/llama2/safety_evaluation/gpt4_eval.py). Please set your OpenAI API key before running the evaluation command:

```bash
python safety_evaluation/gpt4_eval.py --input_file question_output/example.jsonl
```
* `input_file`: a ```jsonl``` file with each line containing the input prompt and the model response.
* The output of the GPT-4 judge will be saved under ```safety_evaluation/gpt4_eval_output```.


## Citation Information

```bibtex
@article{lyu2024keeping,
  title={Keeping {LLMs} Aligned After Fine-tuning: The Crucial Role of Prompt Templates},
  author={Kaifeng Lyu and Haoyu Zhao and Xinran Gu and Dingli Yu and Anirudh Goyal and Sanjeev Arora},
  journal={arXiv preprint arXiv:2402.18540},
  year={2024}
}
```
