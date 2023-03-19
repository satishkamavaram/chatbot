from dataclasses import dataclass
import tensorflow as tf
import io
import os
import json
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from chat_app.com.api.chat.utils.utils import *
import keras.engine


@dataclass
class MODEL_HYPER_PARAMS:
    NUM_WORDS = 10000
    EMBEDDING_DIM = 64
    MAXLEN = 120
    PADDING = 'post'
    OOV_TOKEN = "<OOV>"
    TRAINING_SPLIT = 1.0
    BATCH_SIZE = 100


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.96 and logs.get('accuracy') <= 0.99):
            print("\nReached greater than 96.0% and less than 98.0% accuracy so cancelling training!")
            self.model.stop_training = True


class Model:
    def __init__(self, model: keras.engine.sequential.Sequential,
                 sentence_tokenizer: keras.preprocessing.text.Tokenizer,
                 label_tokenizer: keras.preprocessing.text.Tokenizer):
        self.model = model
        self.sentence_tokenizer = sentence_tokenizer
        self.label_tokenizer = label_tokenizer


cache_model = None


def read_and_pre_process(sentences, labels):
    print(f"There are {len(sentences)} sentences in the dataset.\n")
    print(f"First sentence has {len(sentences[0].split())} words (after removing stopwords).\n")
    print(f"There are {len(labels)} labels in the dataset.\n")
    print(f"The first 5 labels are {labels[:5]}")
    train_sentences, val_sentences, train_labels, val_labels = train_val_split(sentences, labels,
                                                                               MODEL_HYPER_PARAMS.TRAINING_SPLIT)
    print(f"There are {len(train_sentences)} sentences for training.\n")
    print(f"There are {len(train_labels)} labels for training.\n")
    print(f"There are {len(val_sentences)} sentences for validation.\n")
    print(f"There are {len(val_labels)} labels for validation.")
    print(train_labels)
    print(val_labels)
    tokenizer = fit_tokenizer(train_sentences, MODEL_HYPER_PARAMS.NUM_WORDS, MODEL_HYPER_PARAMS.OOV_TOKEN)
    word_index = tokenizer.word_index
    print(f"Vocabulary contains {len(word_index)} words\n")
    print("<OOV> token included in vocabulary" if "<OOV>" in word_index else "<OOV> token NOT included in vocabulary")
    train_padded_seq = seq_and_pad(train_sentences, tokenizer, MODEL_HYPER_PARAMS.PADDING, MODEL_HYPER_PARAMS.MAXLEN)
    val_padded_seq = seq_and_pad(val_sentences, tokenizer, MODEL_HYPER_PARAMS.PADDING, MODEL_HYPER_PARAMS.MAXLEN)
    print(f"Padded training sequences have shape: {train_padded_seq.shape}\n")
    print(f"Padded validation sequences have shape: {val_padded_seq.shape}")
    label_tokenizer, train_label_seq = tokenize_labels(labels, train_labels)
    label_tokenizer, val_label_seq = tokenize_labels(labels, val_labels)
    print(f"First 5 labels of the training set should look like this:\n{train_label_seq[:5]}\n")
    print(f"First 5 labels of the validation set should look like this:\n{val_label_seq[:5]}\n")
    print(f"Tokenized labels of the training set have shape: {train_label_seq.shape}\n")
    print(f"Tokenized labels of the validation set have shape: {val_label_seq.shape}\n")
    return train_padded_seq, val_padded_seq, train_label_seq, val_label_seq, tokenizer, label_tokenizer


def create_model(num_words, embedding_dim, maxlen, total_labels):
    tf.random.set_seed(123)
    print(num_words)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(num_words, embedding_dim, input_length=maxlen),
        # tf.keras.layers.GlobalAveragePooling1D(),
        # tf.keras.layers.Dense(24, activation='relu'),
        # tf.keras.layers.Dense(5, activation='softmax')
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dense(1, activation='sigmoid')
        tf.keras.layers.Dense(total_labels, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def train_model_with_input_file(input_filenmae, epochs: int = 120, save_model_path: str = None):
    sentences, labels = parse_data_from_file(input_filenmae)
    print(sentences)
    model = train_model_sentence_label(sentences, labels, epochs, save_model_path)
    return model


def train_model_sentence_label(sentences, labels, epochs: int = 120, save_model_path: str = None):
    train_padded_seq, val_padded_seq, train_label_seq, val_label_seq, tokenizer, label_tokenizer = read_and_pre_process(
        sentences, labels)
    model = train_model(train_padded_seq, val_padded_seq, train_label_seq, val_label_seq, tokenizer, label_tokenizer,
                        epochs, save_model_path)
    return model


def train_model(train_padded_seq, val_padded_seq, train_label_seq, val_label_seq, sentence_tokenizer, label_tokenizer,
                epochs: int = 120, save_model_path: str = None):
    print(f'label_tokenizer: {label_tokenizer.word_index.items()}')
    print(f'label_tokenizer: {len(label_tokenizer.word_index.items())}')
    total_labels = len(label_tokenizer.word_index.items())
    model = create_model(MODEL_HYPER_PARAMS.NUM_WORDS, MODEL_HYPER_PARAMS.EMBEDDING_DIM, MODEL_HYPER_PARAMS.MAXLEN,
                         total_labels)
    print(epochs)
    print(train_padded_seq)
    print(train_label_seq)
    print(val_padded_seq)
    print(val_label_seq)
    print(len(train_padded_seq))
    callbacks = myCallback()
   # history = model.fit(train_padded_seq, train_label_seq, epochs=epochs, steps_per_epoch=(len(train_padded_seq)/MODEL_HYPER_PARAMS.BATCH_SIZE),
   #                     validation_data=(val_padded_seq, val_label_seq), verbose=2, callbacks=callbacks)
    #history = model.fit(train_padded_seq, train_label_seq, epochs=epochs,
    #                    validation_data=(val_padded_seq, val_label_seq), verbose=2, callbacks=callbacks)
    history = model.fit(train_padded_seq, train_label_seq, epochs=epochs,
                        batch_size=MODEL_HYPER_PARAMS.BATCH_SIZE,
                        validation_data=(val_padded_seq, val_label_seq), verbose=2, callbacks=callbacks)

    acc = history.history['accuracy']
    loss = history.history['loss']
    # global cache_model
    model_obj = Model(model, sentence_tokenizer, label_tokenizer)
    if save_model_path is not None:
        model.save(save_model_path)
        save_tokenizer_to_json(sentence_tokenizer, save_model_path, "tokenizer_sentence.json")
        save_tokenizer_to_json(label_tokenizer, save_model_path, "tokenizer_label.json")
    return model_obj


def load_model(save_model_path: str = None):
    model = tf.keras.models.load_model(save_model_path)
    sentence_tokenizer = load_tokenizer_from_json(save_model_path, "tokenizer_sentence.json")
    label_tokenizer = load_tokenizer_from_json(save_model_path, "tokenizer_label.json")
    return model, sentence_tokenizer, label_tokenizer


def load_saved_model(save_model_path: str = None):
    model = tf.keras.models.load_model(save_model_path)
    sentence_tokenizer = load_tokenizer_from_json(save_model_path, "tokenizer_sentence.json")
    label_tokenizer = load_tokenizer_from_json(save_model_path, "tokenizer_label.json")
    saved_model = Model(model, sentence_tokenizer, label_tokenizer)
    return saved_model


def save_tokenizer_to_json(tokenizer: Tokenizer, save_model_path: str, json_file: str):
    tokenizer_json = tokenizer.to_json()
    with io.open(os.path.join(save_model_path, json_file), 'w', encoding='utf-8') as f:
        f.write(json.dumps(tokenizer_json, ensure_ascii=False))


def load_tokenizer_from_json(save_model_path: str, json_file: str):
    with open(os.path.join(save_model_path, json_file)) as f:
        data = json.load(f)
        return tf.keras.preprocessing.text.tokenizer_from_json(data)


def assign_to_model(model, sentence_tokenizer, label_tokenizer):
    global cache_model
    cache_model = Model(model, sentence_tokenizer, label_tokenizer)


# def predict(save_model_path, input_sentence):
def predict(cache_model, input_sentence):
    print("start")
    # if cache_model is None:
    #    model, sentence_tokenizer, label_tokenizer = load_model(save_model_path)
    #    assign_to_model(model, sentence_tokenizer, label_tokenizer )
    # else:
    model, sentence_tokenizer, label_tokenizer = cache_model.model, cache_model.sentence_tokenizer, cache_model.label_tokenizer

    print("end")
    label_tokenizer_swap = {v: k for k, v in label_tokenizer.word_index.items()}
    print(f'label word index: {label_tokenizer_swap}')
    sentence = remove_stopwords(input_sentence)
    seq_pad_sentence = seq_and_pad([sentence], sentence_tokenizer, MODEL_HYPER_PARAMS.PADDING,
                                   MODEL_HYPER_PARAMS.MAXLEN)
    prediction = model.predict(seq_pad_sentence)
    print(f'Prediction output: {prediction}')
    predicted_index = np.argmax(prediction)
    print(f'Prediction max index: {predicted_index}')
    print(f'Prediction max index value {prediction[0][predicted_index]}')
    # if prediction[0][predicted_index] <= 0.8:
    #     print("Predicted label: I don't understand what you are saying. I am API bot")
    #     return "I don't understand what you are saying. I am API bot"
    # else:
    print(f'Predicted label: {label_tokenizer_swap.get(predicted_index + 1)}')
    return label_tokenizer_swap.get(predicted_index + 1)


if __name__ == '__main__':
    base_path = os.path.dirname(__file__)
    abs_csv_file_path = os.path.join(base_path, 'bot.csv')
    abs_model_file_path = os.path.join(base_path, "my_model")
    train_model_with_input_file(abs_csv_file_path, 120, abs_model_file_path)
