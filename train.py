# paper: https://aclanthology.org/2021.emnlp-main.552/
# reference implementation: https://github.com/princeton-nlp/SimCSE
#
# this implementation only supports Unsup-SimCSE.
# if you want to run the training of Sup-SimCSE, please modify this code yourself.

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from classopt import classopt
from more_itertools import chunked
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

from sts import STSEvaluation

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# classopt is a library for parsing command line arguments in a dataclass style.
# different from argparse, classopt can enjoy the benefits of type hints.
# see: https://github.com/moisutsu/classopt (let's star it!)
@classopt(default_long=True)
class Args:
    model_name: str = "bert-base-uncased"
    # any data set in line-by-line text format can be used.
    # however, it is worth noting that diversity of the dataset is important for SimCSE.
    # see: https://github.com/princeton-nlp/SimCSE/issues/62
    dataset_dir: Path = "./datasets/unsup-simcse"
    sts_dir: Path = "./datasets/sts"
    output_dir: Path = "./outputs"

    # for more detailed hyperparameter settings, see Appendix.A of the paper
    # FYI: SimCSE is not sensitive to batch sizes and learning rates
    batch_size: int = 64
    # the number of epochs is 1 for Unsup-SimCSE, and 3 for Sup-SimCSE in the paper
    epochs: int = 1
    lr: float = 3e-5
    # num_warmup_steps is 0 by default (i.e. no warmup)
    num_warmup_steps: int = 0

    # see Table D.1 of the paper
    temperature: float = 0.05

    # FYI: max_seq_len of reference implementation is 32
    # it seems short, but it is enough for the STS task
    # you should be careful when you apply SimCSE to other tasks that require longer sequences to be handled properly.
    # for other hyperparameters, see Appendix.A of the paper.
    max_seq_len: int = 32

    # FYI: the paper says that the evaluation interval is 250 steps.
    # however, the example training script of official implementation uses 125 steps.
    # this does not seem to be a problem when the number of training steps is large (i.e. batch size is small), as in BERT (batch_size=64),
    # but it may make some difference when the number of steps is small (i.e. batch size is large), as in RoBERTa (batch_size=512).
    # see: https://github.com/princeton-nlp/SimCSE/blob/511c99d4679439c582beb86a0372c04865610b6b/run_unsup_example.sh
    eval_logging_interval: int = 250

    # if you want to use `fp16`, you may encounter some issues.
    # see: https://github.com/princeton-nlp/SimCSE/issues/38#issuecomment-855457923
    device: str = "cuda:0"

    # due to various influences such as implementation and hardware, the same random seed does not always produce the same results.
    # the hyperparameters used in the paper are tuned with a single random seed,
    # so the results may be slightly different from the paper.
    # if you train your own model, you should preferably re-tune the hyperparameters.
    # FYI: https://github.com/princeton-nlp/SimCSE/issues/63
    seed: int = 42


# Reading text line by line is a very simple processing, so we don't need to use a Dataset class actually.
# However we define a dedicated class for future extensibility.
@dataclass
class SimCSEDataset(Dataset):
    path: Path
    data: List[str] = None

    # For simplicity, this dataset class is designed to tokenize text for each loop,
    # but if performance is more important, you should tokenize all text in advance.
    def __post_init__(self):
        self.data = []
        with self.path.open() as f:
            # to prevent whole text into memory at once
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class SimCSEModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        # you can use any models
        self.backbone: PreTrainedModel = AutoModel.from_pretrained(model_name)

        # define additional MLP layer
        # see Section 6.3 of the paper for more details
        # refenrece: https://github.com/princeton-nlp/SimCSE/blob/511c99d4679439c582beb86a0372c04865610b6b/simcse/models.py#L19
        self.hidden_size: int = self.backbone.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor = None,
        # RoBERTa variants don't have token_type_ids, so this argument is optional
        token_type_ids: Tensor = None,
    ) -> Tensor:
        # shape of input_ids: (batch_size, seq_len)
        # shape of attention_mask: (batch_size, seq_len)
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # take representations of [CLS] token
        # we only implement the best performing pooling, [CLS], for simplicity
        # you can easily extend to other poolings (such as mean pooling or max pooling) by edting this line
        # shape of last_hidden_state: (batch_size, seq_len, hidden_size)
        emb = outputs.last_hidden_state[:, 0]

        # original SimCSE uses MLP layer only during training
        # see: Table 6 of the paper
        # this trick is a bit complicated, so you may omit it when training your own model
        if self.training:
            emb = self.dense(emb)
            emb = self.activation(emb)
        # shape of emb: (batch_size, hidden_size)
        return emb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args: Args):
    logging.set_verbosity_error()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model: SimCSEModel = SimCSEModel(args.model_name).to(args.device)

    train_dataset = SimCSEDataset(args.dataset_dir / "train.txt")

    # `collate_fn` is for processing the list of samples to form a batch
    # see: https://discuss.pytorch.org/t/how-to-use-collate-fn/27181
    def collate_fn(batch: List[str]) -> BatchEncoding:
        return tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=args.max_seq_len,
        )

    # see: https://pytorch.org/docs/stable/data.html
    #      https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=True,
        # num_workers and pin_memory are for speeding up training
        num_workers=4,
        pin_memory=True,
        # batch_size varies in the last batch because
        # the last batch size will be the number of remaining samples (i.e. len(train_dataloader) % batch_size)
        # to avoid unstablity of contrastive learning, we drop the last batch
        drop_last=True,
    )

    # FYI: huggingface/transformers' AdamW implementation is deprecated and you should use PyTorch's AdamW instead.
    # see: https://github.com/huggingface/transformers/issues/3407
    #      https://github.com/huggingface/transformers/issues/18757
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    # reference implementation uses a linear scheduler with warmup, which is a default scheduler of transformers' Trainer
    # with num_training_steps = 0 (i.e. no warmup)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        # len(train_dataloader) is the number of steps in one epoch
        num_training_steps=len(train_dataloader) * args.epochs,
    )

    # evaluation class for STS task
    # we use a simple cosine similarity as a semantic similarity
    # and use Spearman's correlation as an evaluation metric
    # see: `sts.py`
    sts = STSEvaluation(sts_dir=args.sts_dir)

    # encode sentences (List[str]) and output embeddings (Tensor)
    # this is for evaluation
    @torch.inference_mode()
    def encode(texts: List[str]) -> torch.Tensor:
        embs = []
        model.eval()
        for text in chunked(texts, args.batch_size * 8):
            batch: BatchEncoding = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            # SimCSE uses MLP layer only during training
            # in this implementation, we use `model.training` to switch between training and evaluation
            emb = model(**batch.to(args.device))
            embs.append(emb.cpu())
        # shape of output: (len(texts), hidden_size)
        return torch.cat(embs, dim=0)

    # evaluation before training
    model.eval()
    best_stsb = sts.dev(encode=encode)
    best_step = 0

    # evaluate the model and store metrics before training
    # this is important to check the appropriateness of training procedure
    print(f"epoch: {0:>3} |\tstep: {0:>6} |\tloss: {' '*9}nan |\tSTSB: {best_stsb:.4f}")
    logs: List[Dict[str, Union[int, float]]] = [
        {
            "epoch": 0,
            "step": best_step,
            "loss": None,
            "stsb": best_stsb,
        }
    ]

    # finally, start training!
    for epoch in range(args.epochs):
        model.train()

        # tqdm makes it easy to visualize how well the training is progressing
        for step, batch in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            dynamic_ncols=True,
        ):
            # transfer batch to the device
            batch: BatchEncoding = batch.to(args.device)
            # if you want to see the actual data, please uncomment the following line.
            # print(batch)
            # and also, if you want to see the actual input strings, please uncomment the following line.
            # print(tokenizer.batch_decode(batch.input_ids, skip_special_tokens=True))

            # simply forward inputs twice!
            # different dropout masks are adapt automatically
            emb1 = model.forward(**batch)
            emb2 = model.forward(**batch)

            # SimCSE training objective:
            #    maximize the similarity between the same sentence
            # => make diagonal elements most similar

            # shape of sim_matrix: (batch_size, batch_size)
            # calculate cosine similarity between all pair of embeddings (n x n)
            sim_matrix = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)
            # FYI: SimCSE is sensitive for the temperature parameter.
            # see Table D.1 of the paper
            sim_matrix = sim_matrix / args.temperature

            # labels := [0, 1, 2, ..., batch_size - 1]
            # labels indicate the index of the diagonal element (i.e. positive examples)
            labels = torch.arange(args.batch_size).long().to(args.device)
            # it may seem strange to use Cross-Entropy Loss here.
            # this is a shorthund of doing SoftMax and maximizing the similarity of diagonal elements
            loss = F.cross_entropy(sim_matrix, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            # for every `args.eval_logging_interval` steps, perform evaluation on STS task and print logs
            if (step + 1) % args.eval_logging_interval == 0 or (step + 1) == len(train_dataloader):
                model.eval()
                # evaluate on the STS-B development set
                stsb_score = sts.dev(encode=encode)

                # you should use the best model for the evaluation to avoid using overfitted model
                # FYI: https://github.com/princeton-nlp/SimCSE/issues/62
                if best_stsb < stsb_score:
                    best_stsb = stsb_score
                    best_step = step + 1
                    # only save the best performing model
                    torch.save(model.state_dict(), args.output_dir / "model.pt")

                # use `tqdm.write` instead of `print` to prevent terminal display corruption
                tqdm.write(
                    f"epoch: {epoch:>3} |\tstep: {step+1:>6} |\tloss: {loss.item():.10f} |\tSTSB: {stsb_score:.4f}"
                )
                logs.append(
                    {
                        "epoch": epoch,
                        "step": step + 1,
                        "loss": loss.item(),
                        "stsb": stsb_score,
                    }
                )
                pd.DataFrame(logs).to_csv(args.output_dir / "logs.csv", index=False)

                # if you want to see the changes of similarity matrix, uncomment the following line
                # tqdm.write(str(sim_matrix))
                model.train()

    # save epochs, steps, losses, and STSB dev scores
    with (args.output_dir / "dev-metrics.json").open("w") as f:
        data = {
            "best-step": best_step,
            "best-stsb": best_stsb,
        }
        json.dump(data, f, indent=2, ensure_ascii=False)

    # load the best model for final evaluation
    if (args.output_dir / "model.pt").exists():
        model.load_state_dict(torch.load(args.output_dir / "model.pt"))
    model.eval().to(args.device)

    sts_metrics = sts(encode=encode)
    with (args.output_dir / "sts-metrics.json").open("w") as f:
        json.dump(sts_metrics, f, indent=2, ensure_ascii=False)

    with (args.output_dir / "config.json").open("w") as f:
        data = {k: v if type(v) in [int, float] else str(v) for k, v in vars(args).items()}
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
