# ============================================================================================
# PROBLEM B5
#
# Build and train a neural network model using the Daily Max Temperature.csv dataset.
# Use MAE as the metrics of your neural network model.
# We provided code for normalizing the data. Please do not change the code.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is downloaded from https://github.com/jbrownlee/Datasets
#
# Desired MAE < 0.2 on the normalized dataset.
# ============================================================================================
import tensorflow as tf
import numpy as np
import  csv
import urllib.request

class mycallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('mae') < 0.17):
            print("\n Target traning sudah tercapai, Berhenti training")
            self.model.stop_training = True
def window_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series,axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size +1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)

def solution_B5():
    data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-max-temperatures.csv"
    urllib.request.urlretrieve(data_url, 'daily-max-temperatures.csv')

    time_step = []
    temps = []

    with open('daily-max-temperatures.csv') as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        next(reader)
        step = 0
        for row in reader:
            temps.append(float(row[1]))
            time_step.append(row[0])
            step+=1

    series = temps

    # Normalization Function, Do Not Change This Code
    min = np.min(series)
    max = np.max(series)
    series -= min
    series /= max
    time = np.array(time_step)

    split_time = 2500

    tine_train = time[:split_time]
    x_train = series[:split_time]

    time_valid = time[split_time:]
    x_valid = series[split_time:]


    window_size = 64
    batch_size = 256
    shuffle_buffer_size = 1000

    train_set = window_dataset(x_train, window_size, batch_size, shuffle_buffer_size)
    # 2500,
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(30,input_shape=[None,1]),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    learning_rate_scheduling = tf.keras.callbacks.LearningRateScheduler(
        lambda  epoch : 1e-8 * 10 ** (epoch/20) # exponential scheduling
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-8, momentum=9e-1)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer,
                  metrics=['mae'])
    model.fit(train_set, epochs=200, callbacks=[learning_rate_scheduling, mycallbacks()])
    return  model

if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B5()
    model.save(r"Model\model_B5.h5")