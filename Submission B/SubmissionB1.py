# =============================================================================
# PROBLEM B1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1]
# Do not use lambda layers in your model.
#
# Please be aware that this is a linear model.
# We will test your model with values in a range as defined in the array to make sure your model is linear.
#
# Desired loss (MSE) < 1e-3
# =============================================================================

import tensorflow as tf
import numpy as np
from tensorflow import keras

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = []):
        if logs['loss'] < 1e-3:
            print("\n Hentikan Training, Sudah memenuhi target")
            self.model.stop_training = True
def solution_B1():
    # Do Not Change This Code
    X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
    y = np.array([5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0], dtype=float)

    # Your Code Here
    # Linear Model
    model = keras.Sequential()
    normalizer = tf.keras.layers.Normalization(axis=None, input_shape = (1,))
    normalizer.adapt(X)
    model.add(normalizer)
    model.add(keras.layers.Dense(32))
    model.add(keras.layers.Dense(1))

    model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    model.fit(X,y, epochs=1000, callbacks=CustomCallback())

    print(model.predict([-2.0, 10.0]))
    return model

if __name__ == "__main__":
    model = solution_B1()
    model.save(r"Model\model_B1.h5")