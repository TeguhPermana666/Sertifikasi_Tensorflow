# ======================================================================================================
# PROBLEM A3
#
# Build a classifier for the Human or Horse Dataset with Transfer Learning.
# The test will expect it to classify binary classes.
# Note that all the layers in the pre-trained model are non-trainable.
# Do not use lambda layers in your model.
#
# The horse-or-human dataset used in this problem is created by Laurence Moroney (laurencemoroney.com).
# Inception_v3, pre-trained model used in this problem is developed by Google.
#
# Desired accuracy and validation_accuracy > 97%.

#  Training Pretrained Model
# =======================================================================================================
import zipfile
import urllib.request
from tensorflow.keras.preprocessing.image import  ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import Adam
import  tensorflow as tf


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('val_accuracy') > 0.97 and logs.get('accuracy') > 0.97):
            print("\nTarget telah dicapai, berhenti training !!!")
            self.model.stop_training = True

def solution_A3():
    # Download Weights classification include to inceptionv3
    inceptionv3 = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    urllib.request.urlretrieve(
        inceptionv3, filename='inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
    )
    local_weights_file = "inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    pre_trained_model = InceptionV3(input_shape=(150,150,3),
                                    include_top=False,
                                    weights=None) # arsitekture
    # load Weights
    pre_trained_model.load_weights(local_weights_file)
    # Get the last layer
    last_layer = pre_trained_model.get_layer('mixed7')

    data_url_1 = "https://github.com/dicodingacademy/assets/releases/download/release-horse-or-human/horse-or-human.zip"
    urllib.request.urlretrieve(data_url_1, filename='horse-or-human.zip')
    local_file = 'horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file, mode='r')
    zip_ref.extractall('data/horse-or-human')
    zip_ref.close()

    data_url_2 = "https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/validation-horse-or-human.zip"
    urllib.request.urlretrieve(data_url_2, filename='validation-horse-or-human.zip')
    local_file = 'validation-horse-or-human.zip'
    zip_ref = zipfile.ZipFile(local_file,mode='r')
    zip_ref.extractall('data/validation-horse-or-human')
    zip_ref.close()

    train_dir = 'data/horse-or-human'
    validation_dir = 'data/validation-horse-or-human'

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    validation_datagen = ImageDataGenerator(rescale=1./255)

    # Resize the image to 150x150
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = (150,150),
        batch_size = 32,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(150,150),
        batch_size=32,
        class_mode='binary'
    )

    # Create arcihitecture model => Bottom layer Based on function rule
    X = layers.Flatten()(last_layer.output)
    X = layers.Dense(units=1024, activation='relu')(X)
    X = layers.Dropout(0.5)(X)
    X = layers.Dense(units=1, activation='sigmoid')(X)
    model = Model(pre_trained_model.input, X)
    callback = myCallback()
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_generator,validation_data=validation_generator, epochs=10,verbose=2, callbacks=callback)
    return model

if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model=solution_A3()
    model.save(r"Model/model_A3.h5")