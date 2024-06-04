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


class profileEngine(Model):
    def __init__(self, model, leakage_model):
        """model: DL model(cnn/mlp)
        leakage_model: a function used for labeling"""
        super().__init__()
        self.model = model
        self.leakage_model = leakage_model

    def train(
        self,
        x_train,
        metadata,
        key_range,
        num_classes,
        hist_acc,
        epochs=300,
        batch_size=100,
    ):
        """
        x_train, metadata: leakages and additional data used for training.
        From the paper (https://tches.iacr.org/index.php/TCHES/article/view/7387/6559), the attack may work when hist_acc= 'accuracy'
        or 'val_accuracy'"""
        self.acc = np.zeros((len(key_range), epochs), dtype=np.double)
        """Trying possible keys"""
        for index, guess in enumerate(key_range):
            print(f"Trying guess = {guess}")
            y_train = np.array([self.leakage_model(i, guess) for i in metadata])
            y = keras.utils.to_categorical(y_train, num_classes)
            self.history = self.model.fit(
                x_train, y, epochs=epochs, batch_size=batch_size, validation_split=0.1
            )
            self.acc[index] = self.history.history[hist_acc]
        return self.acc
