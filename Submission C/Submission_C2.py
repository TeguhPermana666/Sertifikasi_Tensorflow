# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import tensorflow as tf

class customCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get("accuracy") > 0.92 and logs.get("val_accuracy") > 0.92):
            print("\nHentikan Training, target sudah terpenuhi")
            self.model.stop_training = True

def solution_C2():
    # load dataset
    mnist = tf.keras.datasets.mnist
    # Normalize Image
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()

    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0

    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0

    # Define Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2), 2),
        tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2), 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    #Compile Code
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    #Train Model
    model.fit(training_images, training_labels,
              validation_data=(test_images, test_labels),
              epochs=10,verbose=1, callbacks=customCallback())
    return model

# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C2()
    model.save("Model\model_C2.h5")
