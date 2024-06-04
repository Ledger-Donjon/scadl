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



import tensorflow as tf
from keras import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten


class multiLabelEngine(Model):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.leakage_model = leakage_model

    def train(self, x_train, y_train, epochs=300, batch_size=100):
        self.model.fit(
            x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1
        )

    def save_model(self, name):
        self.model.save(name)


class matchEngine(Model):
    def __init__(self, model, leakage_model):
        super().__init__()
        self.model = model
        self.leakage_model = leakage_model

    def match(
        self, x_test, metadata, guess_range, correct_key, step, prob_range=(0, 256)
    ):
        """ 
        x_test, metadata: data used for profiling.
        prob_range depending on the targeted byte
        for ex: k0: (0, 256), k1: (256, 512), k2: (512, 768), .... etc
        """
        rank = []
        number_traces = 0
        x_rank = []
        self.predictions = self.model.predict(x_test)[:, prob_range[0] : prob_range[1]]
        rank_array = np.zeros(guess_range)
        for i in range(0, len(x_test), step):
            chunk = self.predictions[i : i + step]
            chunk_metdata = metadata[i : i + step]
            len_predictions = len(chunk)
            for row in range(len_predictions):
                for guess in range(guess_range):
                    index = self.leakage_model(chunk_metdata[row], guess)
                    if chunk[row, index] != 0:
                        rank_array[guess] += np.log(chunk[row, index])
                    # guess_predictions[row, guess] = self.predictions[row, guess]
            tmp_rank = np.where(sorted(rank_array)[::-1] == rank_array[correct_key])[0][
                0
            ]
            rank.append(tmp_rank)
            number_traces += step
            x_rank.append(number_traces)
        return rank, x_rank
