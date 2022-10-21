# Simple-SimCSE: A simple PyTorch implementation of SimCSE

## Introduction

[SimCSE](https://aclanthology.org/2021.emnlp-main.552/) is one of the most exciting sentence embedding method using contrastive learning.
SimCSE achived state of the art performance, and advance representation learning in NLP greatly.

The concept of SimCSE is very simple in itself (as the title suggests).
However, [the official implementation of SimCSE](https://github.com/princeton-nlp/SimCSE) is abstracted to accommodate a variety of use cases, making the code a bit harder to read.

Of course, the official implementation is great.
However, a simpler implementation would be more helpful in understanding, in particular, it is important for those who are new to research about deep learning or who are just starting out with research about sentence embedding.

Therefore, I implemented a simple version, with minimal abstraction and use of external libraries.

Only using some basic features of [PyTorch](https://github.com/pytorch/pytorch) and [transformers](https://github.com/huggingface/transformers), I developed code to perform fine-tuning of SimCSE from scratch.


## About Implementation


### `download.sh`

### `train.py`

### `sts.py`

### `eval.py`

### Misc



## Instllation

For development, I used [poetry](https://python-poetry.org/), which is the dependency management and packaging tool for Python.

If you use poetry, you can install some necessary packages by following command.

```bash
poetry install
```

Or, in addition, you can install them using `requiments.txt`.
The `requirements.txt` is output by following command.

```bash
poetry export -f requirements.txt --output requirements.txt
```