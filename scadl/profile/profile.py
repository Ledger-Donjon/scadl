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


from ..augmentation import mixup, random_crop
import numpy as np
from keras.models import Model
import keras


class profileEngine:
    """This class is used for normal profiling.
    It takes two argiments: the DL model and the leakage model
    """

    def __init__(self, model: Model, leakage_model):
        super().__init__()
        self.model = model
        self.leakage_model = leakage_model

    def data_augmentation(self, func_aug):
        self.data_aug = func_aug

    def train(
        self,
        x_train: np.ndarray,
        metadata: np.ndarray,
        epochs=300,
        batch_size=100,
        validation_split=0.1,
        data_augmentation=False,
    ):
        """This function is used to train the model
        x_train: poi from leakages
        metadata: the plaintexts, keys, ciphertexts used for profiling
        """
        y_train = np.array([self.leakage_model(i) for i in metadata])
        y_train = keras.utils.to_categorical(y_train, 256)
        if data_augmentation:
            x, y = self.data_aug(x_train, y_train)
        else:
            x, y = x_train, y_train

        self.history = self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
        )

    def save_model(self, name):
        self.model.save(name)


class matchEngine(Model):
    """This class is used for testing the attack after the profiling phase"""

    def __init__(self, model, leakage_model):
        """model: after training the profile model this is fed to this class to test the attack
        leakage_model: The same leakage model used for profiling"""
        super().__init__()
        self.model = model
        self.leakage_model = leakage_model

    def match(self, x_test, metadata, guess_range, correct_key, step):
        """They key rank is implemnted based on the sum of np.log() of the prob"""
        rank = []
        number_traces = 0
        x_rank = []
        self.predictions = self.model.predict(x_test)
        rank_array = np.zeros(guess_range)

        """success rate is calcultaed as shown in https://eprint.iacr.org/2006/139.pdf"""
        for i in range(0, len(x_test), step):
            chunk = self.predictions[i : i + step]
            chunk_metdata = metadata[i : i + step]
            len_predictions = len(chunk)
            for row in range(len_predictions):
                for guess in range(guess_range):
                    index = self.leakage_model(chunk_metdata[row], guess)
                    if chunk[row, index] != 0:
                        rank_array[guess] += np.log2(
                            chunk[row, index]
                        )  # sum of np.log (predictions)
            tmp_rank = np.where(sorted(rank_array)[::-1] == rank_array[correct_key])[0][
                0
            ]
            rank.append(tmp_rank)
            number_traces += step
            x_rank.append(number_traces)

        return np.array(rank), x_rank
