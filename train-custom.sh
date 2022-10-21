# example configurations for custom training
# you can train your own model like this!

poetry run python train.py \
    --model_name roberta-base \
    --batch_size 512 \
    --epochs 3 \
    --lr 1e-5 \
    --temperature 0.1 \
    --max_seq_len 128 \
    --eval_logging_interval 10 \
    --seed 0 \
    --device "cuda:1"