# if you want to perform evaluation only, you can use this snippet ↓

import json
from pathlib import Path
from typing import List

import torch
from classopt import classopt
from more_itertools import chunked
from transformers import AutoTokenizer, logging
from transformers.tokenization_utils import BatchEncoding

from sts import STSEvaluation
from train import SimCSEModel


@classopt(default_long=True)
class Args:
    model_name: str = "bert-base-uncased"
    model_path: Path = "./outputs/model.pt"
    sts_dir: Path = "./datasets/sts"
    output_dir: Path = "./outputs"
    batch_size: int = 512
    device: str = "cuda:0"


def main(args: Args):
    logging.set_verbosity_error()

    model = SimCSEModel(args.model_name)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    model.eval().to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    @torch.inference_mode()
    def encode(texts: List[str]) -> torch.Tensor:
        embs = []
        for text in chunked(texts, args.batch_size):
            batch: BatchEncoding = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            emb = model(**batch.to(args.device), use_mlp=False)
            embs.append(emb.cpu())
        return torch.cat(embs, dim=0)

    evaluation = STSEvaluation(sts_dir=args.sts_dir)
    metrics = evaluation(encode=encode)
    print(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
