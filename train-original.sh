# original configurations described in the paper are used
# see: Table A.1 of Appendix.A


function bert-base-uncased() {
    poetry run python train.py \
        --model_name bert-base-uncased \
        --batch_size 64 \
        --lr 3e-5 \
        --output_dir ./outputs/bert-base-uncased
}

function bert-large-uncased() {
    poetry run python train.py \
        --model_name bert-large-uncased \
        --batch_size 64 \
        --lr 1e-5 \
        --output_dir ./outputs/bert-large-uncased
}


function roberta-base() {
    poetry run python train.py \
        --model_name roberta-base \
        --batch_size 512 \
        --lr 1e-5 \
        --output_dir ./outputs/roberta-base
}


function roberta-large() {
    poetry run python train.py \
        --model_name roberta-large \
        --batch_size 512 \
        --lr 3e-5 \
        --output_dir ./outputs/roberta-large
}

# run training!
# you should change the function name to run different models
bert-base-uncased