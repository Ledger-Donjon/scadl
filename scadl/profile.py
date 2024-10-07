# This file is part of scadl
#
# scadl is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#
# Copyright 2024 Karim ABDELLATIF, PhD, Ledger - karim.abdellatif@ledger.fr

from collections.abc import Callable
from typing import Optional

import keras
import numpy as np
from keras.models import Model
from sklearn.model_selection import train_test_split


class Profile:
    """This class is used for normal profiling.
    It takes two argiments: the DL model and the leakage model
    """

    def __init__(self, model: Model, leakage_model: Callable[[np.ndarray], int]):
        super().__init__()
        self.model = model
        self.leakage_model = leakage_model
        self.data_aug: Optional[
            Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
        ] = None
        self.history = None

    def data_augmentation(
        self,
        func_aug: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    ):
        """to pass the self"""
        self.data_aug = func_aug

    def train(
        self,
        x_train: np.ndarray,
        metadata: np.ndarray,
        guess_range: int,
        epochs: int = 300,
        batch_size: int = 100,
        validation_split: float = 0.1,
        data_augmentation: bool = False,
        verbose: int = 1,
        **kwargs,
    ):
        """This function is used to train the model
        x_train: poi from leakages
        metadata: the plaintexts, keys, ciphertexts used for profiling
        """

        assert self.data_aug is not None

        y_train = np.array([self.leakage_model(m) for m in metadata])
        y_train = keras.utils.to_categorical(y_train, guess_range)
        if data_augmentation:
            x, y = self.data_aug(x_train, y_train)
        else:
            x, y = x_train, y_train

        x_training, x_test, y_training, y_test = train_test_split(
            x, y, test_size=validation_split
        )
        self.history = self.model.fit(
            x_training,
            y_training,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_data=(x_test, y_test),
            **kwargs,
        )

    def save_model(self, name: str):
        """It accepts a str to save the file name"""
        self.model.save(name)


class Match:
    """This class is used for testing the attack after the profiling phase"""

    def __init__(self, model: Model, leakage_model: Callable[[np.ndarray, int], int]):
        """model: after training the profile model this is fed to this class to test the attack
        leakage_model: The same leakage model used for profiling
        """
        super().__init__()
        self.model = model
        self.leakage_model = leakage_model

    def match(
        self,
        x_test: np.ndarray,
        metadata: np.ndarray,
        guess_range: int,
        correct_key: int,
        step: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """They key rank is implemented based on the sum of np.log() of the prob
        success rate is calculated as shown in https://eprint.iacr.org/2006/139.pdf
        """
        predictions = self.model.predict(x_test)

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
                    index = self.leakage_model(metadata_chunk[row], guess)
                    if pred_chunk[row, index] != 0:
                        rank_array[guess] += np.log2(pred_chunk[row, index])
            rank[i] = np.where(sorted(rank_array)[::-1] == rank_array[correct_key])[0][
                0
            ]

            number_traces += step
            x_rank[i] = number_traces

        return rank, x_rank
