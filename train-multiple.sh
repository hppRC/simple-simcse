function bert-base-uncased() {
    for k in {0..49}; do
        poetry run python train.py \
            --model_name bert-base-uncased \
            --batch_size 64 \
            --lr 3e-5 \
            --output_dir ./outputs/bert-base-uncased/$k \
            --device "cuda:0" \
            --seed $k
    done
}

function bert-large-uncased() {
    for k in {0..49}; do
        poetry run python train.py \
            --model_name bert-large-uncased \
            --batch_size 64 \
            --lr 1e-5 \
            --output_dir ./outputs/bert-large-uncased/$k \
            --device "cuda:1" \
            --seed $k
    done
}


function roberta-base() {
    for k in {0..49}; do
        poetry run python train.py \
            --model_name roberta-base \
            --batch_size 512 \
            --lr 1e-5 \
            --output_dir ./outputs/roberta-base/$k \
            --device "cuda:2" \
            --seed $k
    done
}

function roberta-large() {
    for k in {0..49}; do
        poetry run python train.py \
            --model_name roberta-large \
            --batch_size 512 \
            --lr 3e-5 \
            --output_dir ./outputs/roberta-large/$k \
            --device "cuda:3" \
            --seed $k
    done
}

$1