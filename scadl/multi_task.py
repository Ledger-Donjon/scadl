from collections.abc import Callable

import numpy as np


def compute_guessing_entropy(
    predictions: np.ndarray,
    leakage_model: Callable[[np.ndarray, int], int],
    metadata: np.ndarray,
    guess_range: int,
    correct_key: int,
    step: int,
    num_attacks: int,
):
    """Approximate the guessing entropy as defined in https://eprint.iacr.org/2006/139.pdf"""

    assert len(predictions) > 0 and len(predictions) == len(metadata)
    assert correct_key < guess_range
    assert step >= 1
    assert num_attacks >= 1

    sum_rank = np.zeros(len(range(0, len(predictions), step)), dtype=np.uint32)
    permutation = np.arange(len(predictions))
    for _ in range(num_attacks):
        np.random.shuffle(permutation)
        rank, x_rank = compute_rank(
            predictions[permutation],
            leakage_model,
            metadata[permutation],
            guess_range,
            correct_key,
            step,
        )
        sum_rank += rank

    return sum_rank / num_attacks, x_rank


def compute_rank(
    predictions: np.ndarray,
    leakage_model: Callable[[np.ndarray, int], int],
    metadata: np.ndarray,
    guess_range: int,
    correct_key: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the key rank.

    Ref:
    - https://eprint.iacr.org/2006/139.pdf
    """
    assert len(predictions) > 0 and len(predictions) == len(metadata)
    assert correct_key < guess_range
    assert step >= 1

    chunk_starts = range(0, len(predictions), step)
    rank = np.zeros(len(chunk_starts), dtype=np.uint32)
    x_rank = np.zeros(len(chunk_starts), dtype=np.uint32)
    number_traces = 0
    guesses_score = np.zeros(guess_range)
    for i, chunk_start in enumerate(chunk_starts):
        pred_chunk = predictions[chunk_start : chunk_start + step]
        metadata_chunk = metadata[chunk_start : chunk_start + step]
        for row in range(len(pred_chunk)):
            m = np.min(pred_chunk[row, pred_chunk[row] != 0])
            for guess in range(guess_range):
                index = leakage_model(metadata_chunk[row], guess)
                # Avoid NaNs with log
                if pred_chunk[row, index] == 0:
                    guesses_score[guess] += np.log2(m)
                else:
                    guesses_score[guess] += np.log2(pred_chunk[row, index])
        rank[i] = np.where(sorted(guesses_score)[::-1] == guesses_score[correct_key])[
            0
        ][0]

        number_traces += step
        x_rank[i] = number_traces

    return rank, x_rank
