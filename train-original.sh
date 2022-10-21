# original configurations described in the paper
# see: Table A.1 of Appendix.A


# BERT-base
poetry run python train.py \
    --model_name bert-base-uncased \
    --batch_size 64 \
    --lr 3e-5 \
    --output_dir ./outputs/bert-base-uncased


# BERT-large
poetry run python train.py \
    --model_name bert-large-uncased \
    --batch_size 64 \
    --lr 1e-5 \
    --output_dir ./outputs/bert-large-uncased


# RoBERTa-base
poetry run python train.py \
    --model_name roberta-base \
    --batch_size 512 \
    --lr 1e-5 \
    --output_dir ./outputs/roberta-base


# RoBERTa-large
poetry run python train.py \
    --model_name roberta-large \
    --batch_size 512 \
    --lr 3e-5 \
    --output_dir ./outputs/roberta-large