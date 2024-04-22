import torch
import numpy as np
from typing import List, Union, Tuple


def exact_match_rate(
        real_start: Union[List[List[torch.Tensor]], torch.Tensor],
        real_end: Union[List[List[torch.Tensor]], torch.Tensor],
        pred_start: torch.Tensor,
        pred_end: torch.Tensor
) -> Tuple[int, int, float]:

    assert len(real_start) == len(real_end), "real_start and real_end shapes do not match."
    assert pred_start.shape == pred_end.shape, "pred_start and pred_end lengths do not match."
    assert len(real_start) == len(pred_start), \
        f"Datasets mismatch: {len(real_start)} correct labels and {len(pred_start)} predictions were provided."

    correct = 0
    total_indices = len(pred_start) + len(pred_end)
    for i, (pred_start_sample, pred_end_sample) in enumerate(zip(pred_start, pred_end)):
        match_options = []
        # each sample might have j correct possible answers
        for real_start_sample, real_end_sample in zip(real_start[i], real_end[i]):
            matches = 0
            if pred_start_sample == real_start_sample:
                matches += 1
            if pred_end_sample == real_end_sample:
                matches += 1
            match_options.append(matches)
        correct += max(match_options)
    match_rate = correct / total_indices
    return correct, total_indices, match_rate


def f1_score(
        real_start: Union[List[List[torch.Tensor]], torch.Tensor],
        real_end: Union[List[List[torch.Tensor]], torch.Tensor],
        pred_start: torch.Tensor,
        pred_end: torch.Tensor
) -> Tuple[List[float], float]:

    all_f1 = []
    for i, (pred_start_sample, pred_end_sample) in enumerate(zip(pred_start, pred_end)):

        pred_indices = set(range(pred_start_sample, pred_end_sample + 1))

        f1_options = []
        # each sample might have j correct possible answers
        for real_start_sample, real_end_sample in zip(real_start[i], real_end[i]):

            real_indices = set(range(real_start_sample, real_end_sample + 1))
            correctly_pred_indices = real_indices.intersection(pred_indices)
            if correctly_pred_indices == set():
                f1_options.append(0)
                continue  # f1 is 0 if there's no overlap. Loop cannot continue to avoid division by zero error.

            precision = len(correctly_pred_indices) / len(pred_indices)
            recall = len(correctly_pred_indices) / len(real_indices)
            f1_sample = (2 * precision * recall) / (precision + recall)
            f1_options.append(f1_sample)
        all_f1.append(float(max(f1_options)))  # float() just so we avoid a mix of int and float in the list

    average_f1 = np.mean(all_f1)
    return all_f1, average_f1


