from pathlib import Path
from typing import Callable, Dict, List, Union

import torch
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances
from torch import Tensor
from tqdm import tqdm

# each STS dataset have different format, so we need to handle them separately
# each STS dataset have subsets, and they are saved in different files
# this module provides a unified interface to access the datasets


class STSEvaluatorBase:
    def __init__(
        self,
        sentences1: List[str],
        sentences2: List[str],
        scores: List[float],
    ):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        assert len(self.sentences1) == len(self.sentences2) == len(self.scores)

    def __call__(self, encode: Callable[[List[str]], Tensor]) -> float:
        embeddings1 = encode(self.sentences1)
        embeddings2 = encode(self.sentences2)
        # you can use any similarity function you want ↓
        cosine_scores = 1 - paired_cosine_distances(embeddings1, embeddings2)
        spearman = float(spearmanr(self.scores, cosine_scores)[0]) * 100

        return spearman


class SICKEvaluator(STSEvaluatorBase):
    # Title: A SICK cure for the evaluation of compositional distributional semantic models
    # URL: https://aclanthology.org/L14-1314/

    def __init__(self, sts_dir: Path):
        sentences1, sentences2, scores = [], [], []

        with (sts_dir / "sick/SICK_test_annotated.txt").open() as f:
            _ = next(f)
            for line in f:
                _, sentence1, sentence2, score, *_ = line.strip().split("\t")
                sentences1.append(sentence1)
                sentences2.append(sentence2)
                scores.append(float(score))

        super().__init__(sentences1, sentences2, scores)


class STSBDevEvaluator(STSEvaluatorBase):
    def __init__(self, sts_dir: Path):
        sentences1, sentences2, scores = [], [], []
        with (sts_dir / "stsb/sts-dev.csv").open() as f:
            for line in f:
                _, _, _, _, score, sentence1, sentence2, *_ = line.strip().split("\t")
                sentences1.append(sentence1)
                sentences2.append(sentence2)
                scores.append(float(score))

        super().__init__(sentences1, sentences2, scores)


class STSBEvaluator(STSEvaluatorBase):
    # Title: SemEval-2017 Task 1: Semantic Textual Similarity Multilingual and Crosslingual Focused Evaluation
    # URL: https://aclanthology.org/S17-2001/

    def __init__(self, sts_dir: Path):
        sentences1, sentences2, scores = [], [], []
        with (sts_dir / "stsb/sts-test.csv").open() as f:
            for line in f:
                _, _, _, _, score, sentence1, sentence2, *_ = line.strip().split("\t")
                sentences1.append(sentence1)
                sentences2.append(sentence2)
                scores.append(float(score))

        super().__init__(sentences1, sentences2, scores)


class STS16Evaluator(STSEvaluatorBase):
    # Title: SemEval-2016 Task 1: Semantic Textual Similarity, Monolingual and Cross-Lingual Evaluation
    # URL: https://aclanthology.org/S16-1081/

    SUBSETS = [
        "answer-answer",
        "headlines",
        "plagiarism",
        "postediting",
        "question-question",
    ]

    def __init__(self, sts_dir: Path):
        sentences1, sentences2, scores = [], [], []

        for subset in self.SUBSETS:
            with (sts_dir / f"sts16/STS2016.gs.{subset}.txt").open() as gs, (
                sts_dir / f"sts16/STS2016.input.{subset}.txt"
            ).open() as f:

                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores)


class STS15Evaluator(STSEvaluatorBase):
    # Title: SemEval-2015 Task 2: Semantic Textual Similarity, English, Spanish and Pilot on Interpretability
    # URL: https://aclanthology.org/S15-2045/

    SUBSETS = [
        "answers-forums",
        "answers-students",
        "belief",
        "headlines",
        "images",
    ]

    def __init__(self, sts_dir: Path):
        sentences1, sentences2, scores = [], [], []

        for subset in self.SUBSETS:
            with (sts_dir / f"sts15/STS.gs.{subset}.txt").open() as gs, (
                sts_dir / f"sts15/STS.input.{subset}.txt"
            ).open() as f:
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores)


class STS14Evaluator(STSEvaluatorBase):
    # Title: SemEval-2014 Task 10: Multilingual Semantic Textual Similarity
    # URL: https://aclanthology.org/S14-2010/

    SUBSETS = [
        "deft-forum",
        "deft-news",
        "headlines",
        "images",
        "OnWN",
        "tweet-news",
    ]

    def __init__(self, sts_dir: Path):
        sentences1, sentences2, scores = [], [], []

        for subset in self.SUBSETS:
            with (sts_dir / f"sts14/STS.gs.{subset}.txt").open() as gs, (
                sts_dir / f"sts14/STS.input.{subset}.txt"
            ).open() as f:

                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores)


class STS13Evaluator(STSEvaluatorBase):
    # Title: *SEM 2013 shared task: Semantic Textual Similarity
    # URL: https://aclanthology.org/S13-1004/

    SUBSETS = ["FNWN", "headlines", "OnWN"]

    # STS13 here does not contain the "SMT" subtask due to LICENSE issue
    def __init__(self, sts_dir: Path):
        sentences1, sentences2, scores = [], [], []

        for subset in self.SUBSETS:
            with (sts_dir / f"sts13/STS.gs.{subset}.txt").open() as gs, (
                sts_dir / f"sts13/STS.input.{subset}.txt"
            ).open() as f:
                for line_input, line_gs, *_ in zip(f, gs):
                    sentence1, sentence2 = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores)


class STS12Evaluator(STSEvaluatorBase):
    # Title: SemEval-2012 Task 6: A Pilot on Semantic Textual Similarity
    # URL: https://aclanthology.org/S12-1051/

    SUBSETS = [
        "MSRpar",
        "MSRvid",
        "SMTeuroparl",
        "surprise.OnWN",
        "surprise.SMTnews",
    ]

    def __init__(self, sts_dir: Path):
        sentences1, sentences2, scores = [], [], []

        for subset in self.SUBSETS:
            with (sts_dir / f"sts12/STS.gs.{subset}.txt").open() as gs, (
                sts_dir / f"sts12/STS.input.{subset}.txt"
            ).open() as f:
                for line_input, line_gs in zip(f, gs):
                    sentence1, sentence2, *_ = line_input.strip().split("\t")
                    if line_gs.strip() == "":
                        continue
                    sentences1.append(sentence1)
                    sentences2.append(sentence2)
                    scores.append(float(line_gs.strip()))

        super().__init__(sentences1, sentences2, scores)


class STSEvaluation:
    def __init__(self, sts_dir: Union[str, Path]):
        sts_dir = Path(sts_dir)
        self.sts_evaluators = {
            "sts12": STS12Evaluator(sts_dir=sts_dir),
            "sts13": STS13Evaluator(sts_dir=sts_dir),
            "sts14": STS14Evaluator(sts_dir=sts_dir),
            "sts15": STS15Evaluator(sts_dir=sts_dir),
            "sts16": STS16Evaluator(sts_dir=sts_dir),
            "stsb": STSBEvaluator(sts_dir=sts_dir),
            "sick": SICKEvaluator(sts_dir=sts_dir),
        }
        self.dev_evaluator = STSBDevEvaluator(sts_dir=sts_dir)

    @torch.inference_mode()
    def __call__(
        self,
        encode: Callable[[List[str]], Tensor],
        progress_bar: bool = True,
    ) -> Dict[str, float]:

        results = {}
        if progress_bar:
            iterator = tqdm(
                list(self.sts_evaluators.items()),
                dynamic_ncols=True,
                leave=False,
            )
        else:
            iterator = list(self.sts_evaluators.items())

        for name, evaluator in iterator:
            results[name] = evaluator(encode=encode)

        results["avg"] = sum(results.values()) / len(results)
        return results

    @torch.inference_mode()
    def dev(
        self,
        encode: Callable[[List[str]], Tensor],
    ) -> float:
        return self.dev_evaluator(encode=encode)
