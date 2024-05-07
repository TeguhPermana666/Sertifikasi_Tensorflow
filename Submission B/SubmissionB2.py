# =============================================================================
# PROBLEM B2
#
# Build a classifier for the Fashion MNIST dataset.
# The test will expect it to classify 10 classes.
# The input shape should be 28x28 monochrome. Do not resize the data.
# Your input layer should accept (28, 28) as the input shape.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 83%
# =============================================================================

import tensorflow as tf

class Callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if (logs.get("accuracy") > 0.85 and logs.get("val_accuracy") > 0.85):
            print("\n Hentikan Training, hasil sudah tercapai")
            self.model.stop_training = True


def solution_B2():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # Normalize the image
    (training_images, training_labels), (testing_images, testing_labels) = fashion_mnist.load_data()

    training_images.reshape(60000,28,28,1)
    training_images = training_images / 255.0

    testing_images.reshape(10000,28,28,1)
    testing_images = testing_images / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=10, activation='softmax')
    ])


    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_images, training_labels, epochs=1000,
              validation_data=(testing_images,testing_labels),
              verbose=1, callbacks=Callback())

    return model

if __name__ == "__main__":
    model = solution_B2()
    model.save(r"Model\model_B2.h5")