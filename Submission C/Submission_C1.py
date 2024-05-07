# =============================================================================
# PROBLEM C1
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
# Desired loss (MSE) < 1e-4
# =============================================================================


import  tensorflow as tf
import  urllib.request
import  numpy as np

class customCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # if metrics we can directly -> ('mse'),
        # if loss -> ('loss')
        if(logs.get("loss") < 1e-4):
            print("\n Hentikan Training ketika sudah memenuhi target!!")
            self.model.stop_training = True

def solution_C1():
    # DO NOT CHANGE THIS CODE
    X = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    Y = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
    print(X.shape)
    model = tf.keras.Sequential([ #Simple linear layes cant used keras.models, just direct used sequential method
        tf.keras.layers.Dense(1, input_shape=[1])
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
                  loss='mse')
    model.fit(X,Y, epochs=1000, callbacks=customCallback())
    print(model.predict([-2.0, 10.0]))
    return model

# The code below is to save your model as a .h5 file
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C1()
    model.save(r"Model\model_C1.h5")
