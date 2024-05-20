import tensorflow as tf
from keras import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten
import keras
class profileEngine(Model):
    def __init__(self, model, leakage_model):
        super().__init__()
        self.model = model
        self.leakage_model = leakage_model

    def train(self, x_train, metadata, key_range, epochs=300, batch_size=100):
        self.acc = np.zeros((len(key_range), epochs), dtype=np.double)
        for guess in range(len(key_range)):
            print(f"Trying guess = {guess}")
            y_train = np.array([self.leakage_model(i, guess) for i in metadata]) 
            y = keras.utils.to_categorical(y_train, 2)
            self.history = self.model.fit(x_train, y, epochs=epochs, batch_size=batch_size, validation_split=0.15)
            self.acc[guess] = self.history.history['val_accuracy']
        return self.acc

        



    

        




        



        


    







