# ========================================================================================
# PROBLEM B3
#
# Build a CNN based classifier for Rock-Paper-Scissors dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
#
# Desired accuracy AND validation_accuracy > 83%
# ========================================================================================

import tensorflow as tf
import os
import zipfile
import urllib.request
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > 0.85 and logs.get('val_accuracy') > 0.85):
            print('\n Hentikan Training ketika sudah memenuhi target.')
            self.model.stop_training = True

def solution_B3():
    data_url = "https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip"
    urllib.request.urlretrieve(data_url, filename='rps.zip')
    local_file = 'rps.zip'
    zip_ref = zipfile.ZipFile(local_file, mode='r')
    zip_ref.extractall('data/')
    zip_ref.close()

    # Image Size should be 150x150
    # Make sure used "categorical"
    TRAINING_DIR = 'data/rps/'
    training_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        rotation_range=20,
        validation_split=0.2
    )
    train_generator = training_datagen.flow_from_directory(
        TRAINING_DIR,
        target_size=(150,150),
        color_mode='rgb',
        class_mode='categorical',
        subset='training'
    )

    # validation_datagen = ImageDataGenerator(rescale=1./255) -> tidak ada directory
    validation_generator = training_datagen.flow_from_directory(
        TRAINING_DIR, #Actually we used the validation_dir
        target_size=(150,150),
        color_mode='rgb',
        class_mode='categorical',
        subset='validation'
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(3,3), activation='relu', input_shape=(150,150,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy',#for not directly label
                  optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3), metrics=['accuracy'])
    model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=CustomCallback(),
        verbose=1)
    return model

if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_B3()
    model.save(r"Model\model_B3.h5")