from collections.abc import Callable

import numpy as np


def compute_rank(
    predictions: np.ndarray,
    leakage_model: Callable[[np.ndarray, int], int],
    x_test: np.ndarray,
    metadata: np.ndarray,
    guess_range: int,
    correct_key: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """They key rank is implemented based on the sum of np.log() of the prob
    success rate is calculated as shown in https://eprint.iacr.org/2006/139.pdf
    """
    chunk_starts = range(0, len(x_test), step)
    rank = np.zeros(len(chunk_starts), dtype=np.uint32)
    x_rank = np.zeros(len(chunk_starts), dtype=np.uint32)
    number_traces = 0
    rank_array = np.zeros(guess_range)
    for i, chunk_start in enumerate(chunk_starts):
        pred_chunk = predictions[chunk_start : chunk_start + step]
        metadata_chunk = metadata[chunk_start : chunk_start + step]
        for row in range(len(pred_chunk)):
            for guess in range(guess_range):
                index = leakage_model(metadata_chunk[row], guess)
                if pred_chunk[row, index] != 0:
                    rank_array[guess] += np.log2(pred_chunk[row, index])
        rank[i] = np.where(sorted(rank_array)[::-1] == rank_array[correct_key])[0][0]

        number_traces += step
        x_rank[i] = number_traces

    return rank, x_rank
