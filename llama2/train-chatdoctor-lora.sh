PORT=$((25090 + $RANDOM % 1024))
SEED=$(($RANDOM << 15 | $RANDOM))
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 torchrun --nnodes 1 --nproc_per_node 8 finetuning.py \
    --batch_size_training 16 --lr 0.00002 --weight_decay 0 \
    --num_epochs 3 \
    --dataset medical_dataset \
    --optimizer AdamW \
    --scheduler cosine \
    --warmup_ratio 0.03 \
    --medical_dataset.prompt chatdoctor \
    --enable_fsdp \
    --model_name meta-llama/Llama-2-7b-chat-hf --pure_bf16 \
    --dist_checkpoint_root_folder finetuned_models/ \
    --dist_checkpoint_folder chatdoctor \
    --gradient_accumulation_steps 1 \
    --seed $SEED --save_dir .



