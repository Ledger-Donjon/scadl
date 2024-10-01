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

import numpy as np
from keras.models import Model


class MultiLabelProfile:
    """This class is used for multi-label classification"""

    def __init__(self, model: Model):
        super().__init__()
        self.model = model
        self.history = None

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 300,
        batch_size: int = 100,
        validation_split: float = 0.1,
        **kwargs,
    ):
        """This function accepts
        x_train: np.array,
        y_train: np.array,
        """
        self.history = self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            **kwargs,
        )

    def save_model(self, name: str):
        """It takes a string name and saves the model"""
        self.model.save(name)


class MatchMultiLabel:
    """This class is used for testing the attack"""

    def __init__(self, model: Model, leakage_model: Callable):
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
        prob_range: tuple[int, int] = (0, 256),
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        x_test, metadata: data used for profiling.
        prob_range depending on the targeted byte
        for ex: k0: (0, 256), k1: (256, 512), k2: (512, 768), .... etc
        """
        predictions = self.model.predict(x_test)[:, prob_range[0] : prob_range[1]]

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
                        rank_array[guess] += np.log(pred_chunk[row, index])
            rank[i] = np.where(sorted(rank_array)[::-1] == rank_array[correct_key])[0][
                0
            ]

            number_traces += step
            x_rank[i] = number_traces

        return rank, x_rank
