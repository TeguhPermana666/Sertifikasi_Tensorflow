# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import  pad_sequences
import tensorflow as tf
import pandas as pd
import numpy as np

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get("accuracy") > 0.93 and logs.get("val_accuracy") > 0.93):
            print(("\nHentikan Training ketika sudah memenuhi target"))
            self.model.stop_training = True

def train_test_split(features,labels, test_size=0.2, random_state=None):
    # Combine features and labels
    dataset = tf.data.Dataset.from_tensor_slices((features,labels))
    # Shuffle dataset
    if random_state is not None:
        tf.random.set_seed(random_state)
    dataset = dataset.shuffle(buffer_size=len(features), seed= random_state)
    # split dataset
    test_size = int(len(features) * test_size)
    train_dataset = dataset.skip(test_size)
    test_dataset = dataset.take(test_size)

    return train_dataset,test_dataset

def solution_B4():
    # Collect dataset from link dataset
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    sentences = []
    labels = []
    for index, row in bbc.iterrows():
        labels.append(row[0])
        sentences.append(row[1])

    training_size = int(len(sentences) * training_portion)
    # Split based on training size
    training_sentences = sentences[:training_size]
    training_labels = labels[:training_size]

    validation_sentences = sentences[training_size:]
    validation_labels = labels[training_size:]

    # Fit tokenizer with training data
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    training_padded_sequences = pad_sequences(training_sequences, padding=padding_type,
                                             maxlen=max_length, truncating=trunc_type)
    validation_sequences = tokenizer.texts_to_sequences(validation_sentences)
    validation_padded_sequences = pad_sequences(validation_sequences, padding=padding_type,
                                               maxlen= max_length, truncating=trunc_type)

    # Also can did some Tokenizer to encode a labels
    label_tokenizer = Tokenizer()
    label_tokenizer.fit_on_texts(labels)
    labels_word_index = label_tokenizer.word_index

    training_labels_sequences = label_tokenizer.texts_to_sequences(training_labels)
    training_labels_sequences = np.array(training_labels_sequences)

    validation_labels_sequences = label_tokenizer.texts_to_sequences(validation_labels)
    validation_labels_sequences = np.array(validation_labels_sequences)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length= max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
                  optimizer= tf.keras.optimizers.Adam(learning_rate=0.003),
                  metrics=['accuracy']
                  )

    model.fit(
        training_padded_sequences,
        training_labels_sequences,
        epochs=200,
        callbacks=CustomCallback(),
        validation_data=(
            validation_padded_sequences,
            validation_labels_sequences
        ),
        verbose = 2
    )
    return model
if __name__ == '__main__':
    model = solution_B4()
    model.save(r"Model\model_B4.h5")