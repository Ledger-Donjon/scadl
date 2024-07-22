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


import numpy as np
from keras.models import Model
import keras
import tensorflow as tf


class NonProfile:
    """This class is used for Non-profiling DL attacks proposed in https://eprint.iacr.org/2018/196.pdf"""

    def __init__(self, leakage_model):
        """It takes a model and a leakagae_model function"""
        # super().__init__()
        self.leakage_model = leakage_model
        self.acc = None
        self.history = None

    def train(
        self,
        model: Model,
        x_train: np.ndarray,
        metadata: np.ndarray,
        guess: int,
        num_classes: int,
        hist_acc: str,
        epochs=300,
        batch_size=100,
    ) -> np.ndarray:
        """
        x_train, metadata: leakages and additional data used for training.
        From the paper (https://tches.iacr.org/index.php/TCHES/article/view/7387/6559), the attack may work when hist_acc= 'accuracy'
        or 'val_accuracy'"""
        y_train = np.array([self.leakage_model(i, guess) for i in metadata])
        y = keras.utils.to_categorical(y_train, num_classes)
        self.acc = model.fit(
            x=x_train,
            y=y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0,
        ).history[hist_acc]
        return self.acc
