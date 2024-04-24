# ==========================================================================================================
# PROBLEM A4
#
# Build and train a binary classifier for the IMDB review dataset.
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in http://ai.stanford.edu/~amaas/data/sentiment/
#
# Desired accuracy and validation_accuracy > 83%
# ===========================================================================================================

import  tensorflow as tf
import  tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.sequence import  pad_sequences
# For insert the padding of data (beggining value of matriks)
from tensorflow.keras.preprocessing.text import Tokenizer

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('val_accuracy') > 0.85 and logs.get('accuracy') > 0.85):
            print("\n Target telah dicapai, berhenti training !!!")
            self.model.stop_training = True

def solution_A4():
    imdb, info, = tfds.load(name="imdb_reviews", with_info=True, as_supervised=True)
    train_data, test_data = imdb['train'], imdb['test']

    training_sentences = []
    testing_sentences  = []
    training_labels = []
    testing_labels = []

    for s,l in train_data:
        training_sentences.append(s.numpy().decode('utf8'))
        training_labels.append((l.numpy()))
    for s,l in test_data:
        testing_sentences.append(s.numpy().decode('utf8'))
        testing_labels.append(l.numpy())

    vocab_size = 10000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    oov_took = '<OOV>'

    # Fit the tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_took)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded =  pad_sequences(training_sequences, maxlen=max_length, truncating=trunc_type)

    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_sequences, maxlen=max_length)

    training_labels_final = np.array(training_labels)
    testing_labels_final = np.array(testing_labels)

    reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

    def decode_review(text):
        return ' '.join([reverse_word_index.get(i, '?') for i in text])

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=8, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='sigmoid') # Binary Classification
    ])
    callback = myCallback()
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'],)
    model.fit(training_padded, training_labels_final, batch_size=128, epochs=10,
              validation_data=(
                  testing_padded, testing_labels_final
              ), callbacks=callback)

    return model

if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_A4()
    model.save(r"Model/model_A4.h5")

