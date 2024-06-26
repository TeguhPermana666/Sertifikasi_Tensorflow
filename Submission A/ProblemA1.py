# =================================================================================
# PROBLEM A1
#
# Given two arrays, train a neural network model to match the X to the Y.
# Predict the model with new values of X [-2.0, 10.0]
# We provide the model prediction, do not change the code.
#
# The test infrastructure expects a trained model that accepts
# an input shape of [1].
# Do not use lambda layers in your model.
#
# Please be aware that this is a linear model.
# We will test your model with values in a range as defined in the array to make sure your model is linear.
#
# Desired loss (MSE) < 1e-4
#  Training basic model for predictions
# =================================================================================

import numpy as np
import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras import  layers

def solution_A1():
    # Dont change this code
    X = np.array([-4.0, -3.0, -2.0, -1.0, 0.0, 1.0,
                 2.0, 3.0, 4.0, 5.0], dtype=float)
    y = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0,
                 12.0, 13.0, 14.0, ], dtype=float)
    # Create Model
    model = keras.Sequential([
        layers.Dense(units=1, input_shape=[1]) # Simple Linear Layer
    ])
    # Compile model
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # Train the model
    model.fit(X,y, epochs=1000, verbose=0)
    mse = model.evaluate(X,y,verbose=0)
    print("Mean Squared Error:",mse<1e-4)
    print(model.predict([-2.0, 10.0]))
    return model

if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_A1()
    model.save(r"Model\model_A1.h5")