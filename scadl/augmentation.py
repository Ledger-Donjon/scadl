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


class Mixup:
    """This class used for data augmentation
    proposed in https://eprint.iacr.org/2021/328.pdf"""

    def generate(self, x_train: np, y_train: np, ratio: float, alpha=0.2):
        """It taked x_train, y_train, which are leakages and labels"""
        len_augmented_data = int(ratio * len(x_train))
        augmented_data = np.zeros((len_augmented_data, x_train.shape[1]))
        augmented_labels = np.zeros((len_augmented_data, y_train.shape[1]))
        for i in range(len_augmented_data):
            # lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
            lam = np.random.beta(alpha, alpha)
            random_index = np.random.randint(x_train.shape[0] - 1)
            augmented_data[i] = (lam * x_train[random_index]) + (
                (1 - lam) * x_train[random_index + 1]
            )
            augmented_labels[i] = (lam * y_train[random_index]) + (
                (1 - lam) * y_train[random_index + 1]
            )
        return np.concatenate((x_train, augmented_data), axis=0), np.concatenate(
            (y_train, augmented_labels), axis=0
        )


class RandomCrop:
    """A data augmentation technique shown in
    https://blog.roboflow.com/why-and-how-to-implement-random-crop-data-augmentation/"""

    def generate(self, x_train: np, y_train: np, ratio: float, window):
        """It taked x_train, y_train, ratio which are leakages, labels, and data increase ratio"""
        len_augmented_data = int(ratio * len(x_train))
        augmented_data = np.zeros((len_augmented_data, x_train.shape[1]))
        augmented_labels = np.zeros((len_augmented_data, y_train.shape[1]))
        for i in range(len_augmented_data):
            random_index = np.random.randint(x_train.shape[0])
            sample_trace = x_train[random_index]
            random_window = np.random.randint(x_train.shape[1] - window)
            sample_trace[random_window : random_window + window] = 0
            augmented_data[i] = sample_trace
            augmented_labels[i] = y_train[random_index]
        return np.concatenate((x_train, augmented_data), axis=0), np.concatenate(
            (y_train, augmented_labels), axis=0
        )


if __name__ == "__main__":
    leakages = np.random.randint(10, size=(10, 10))
    labels = np.random.randint(3, size=(10, 5))
    mixup = Mixup()
    x, y = mixup.generate(x_train=leakages, y_train=labels, ratio=0.5)
    print("result of mixup")
    print(f"x={x}, y={y}")
    random_crop = RandomCrop()
    x, y = random_crop.generate(x_train=leakages, y_train=labels, ratio=0.5, window=5)
    print("result of random crop")
    print(f"x={x}, y={y}")
