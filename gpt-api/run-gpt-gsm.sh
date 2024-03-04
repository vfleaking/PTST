SEED=$(($RANDOM << 15 | $RANDOM))
export OPENAI_API_KEY=sk-xxxx
python finetuning.py --seed $SEED \
    --data gsm8k --use_hf_data True \
    --data_prompt_key question --data_answer_key answer\
    --template alpaca \
    --n_epochs 1 --wandb online