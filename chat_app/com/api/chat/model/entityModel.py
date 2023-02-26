import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
#from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
#from tensorflow.keras import Model, Input
#from tensorflow.keras.layers import LSTM, Embedding, Dense
#from tensorflow.keras.layers import TimeDistributed, SpatialDropout1D, Bidirectional
import keras.engine
import io
import os
import json
from dataclasses import dataclass


@dataclass
class MODEL_HYPER_PARAMS:
    MAXLEN = 50
    PADDING = 'post'
    BATCH_SIZE = 1

class SentenceGetter(object):
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w.lower(), t) for w, t in zip(s["Word"].values.tolist(),
                                                           #s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') >= 1.0 and logs.get('accuracy') <= 1.0):
            print("\nReached 1.0% accuracy so cancelling training!")
            self.model.stop_training = True

class Model:
    def __init__(self, model: keras.engine.sequential.Sequential,
                 word2idx: dict, tag2idx: dict, words: list, tags: list):
        self.model = model
        self.word2idx = word2idx
        self.tag2idx = tag2idx
        self.words = words
        self.tags = tags


def parse_data_from_file(input_filenmae):
    data = pd.read_csv(input_filenmae)
    data = data.fillna(method="ffill")
    data.head(20)
    print("Unique words in corpus:", data['Word'].nunique())
    print("Unique tags in corpus:", data['Tag'].nunique())
    words = list(set(data["Word"].values))
    words.append("ENDPAD")
    num_words = len(words)
    print(num_words)
    tags = list(set(data["Tag"].values))
    num_tags = len(tags)
    print(num_tags)
    return data, words, tags


def train_model_with_input_file(input_filenmae, epochs: int = 500, save_model_path: str = None):
    data, words, tags = parse_data_from_file(input_filenmae)
    num_words = len(words)
    getter = SentenceGetter(data)
    sentences = getter.sentences
    print(sentences)
    word2idx = {w: i + 1 for i, w in enumerate(words)}
    tag2idx = {t: i for i, t in enumerate(tags)}
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=MODEL_HYPER_PARAMS.MAXLEN, sequences=X, padding="post", value=num_words - 1)

    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=MODEL_HYPER_PARAMS.MAXLEN, sequences=y, padding="post", value=tag2idx["0"])
    model = train_model(X, y, word2idx, tag2idx, words, tags, save_model_path, epochs)
    return model

def create_model(num_words,num_tags):
    model = tf.keras.Sequential([
        # tf.keras.Input(shape=(max_len)),
        # tf.keras.layers.Embedding(num_words, output_dim=max_len, input_length=max_len),
        # tf.keras.layers.SpatialDropout1D(0.1),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_tags, return_sequences=True)),

        # tf.keras.layers.Embedding(num_words, 50, input_length=50),
        # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        # tf.keras.layers.Dropout(0.5),
        # tf.keras.layers.Dense(64, activation='relu'),
        # tf.keras.layers.Dense(num_tags, activation='softmax')

        tf.keras.Input(shape=(MODEL_HYPER_PARAMS.MAXLEN)),
        tf.keras.layers.Embedding(num_words, output_dim=MODEL_HYPER_PARAMS.MAXLEN, input_length=MODEL_HYPER_PARAMS.MAXLEN),
        tf.keras.layers.SpatialDropout1D(0.1),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),  # num_tags or 64 or 32
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_tags, activation='softmax')

    ])
    model.summary()

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  # loss = "categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def train_model(X, y, word2idx: dict, tag2idx: dict, words:list, tags: list,
                save_model_path: str,epochs: int = 500):
    model = create_model(len(words),len(tags))
    callbacks = myCallback()
    model.fit(
        x=X,
        y=y,
        batch_size=MODEL_HYPER_PARAMS.BATCH_SIZE,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    # global cache_model
    model_obj = Model(model, word2idx, tag2idx, words, tags)
    if save_model_path is not None:
        model.save(save_model_path)
        save_json_to_file(word2idx, save_model_path, "word2idx.json")
        save_json_to_file(tag2idx, save_model_path, "tag2idx.json")
        save_to_file(",".join(words), save_model_path, "words.txt")
        save_to_file(",".join(tags), save_model_path, "tags.txt")

    return model_obj

def save_json_to_file(data: dict, save_model_path: str, filename: str):
    with io.open(os.path.join(save_model_path, filename), 'w', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False))

def save_to_file(data: dict, save_model_path: str, filename: str):
    with io.open(os.path.join(save_model_path, filename), 'w', encoding='utf-8') as f:
        f.write(data)

def load_json_from_json(save_model_path: str, filename: str):
    with open(os.path.join(save_model_path, filename)) as f:
        return json.load(f)

def load_from_file(save_model_path: str, filename: str):
    with open(os.path.join(save_model_path, filename)) as f:
        return f.read()

def load_saved_model(save_model_path: str = None):
    model = tf.keras.models.load_model(save_model_path)
    word2idx = load_json_from_json(save_model_path, "word2idx.json")
    tag2idx = load_json_from_json(save_model_path, "tag2idx.json")
    words = load_from_file(save_model_path, "words.txt").split(",")
    tags = load_from_file(save_model_path, "tags.txt").split(",")
    saved_model = Model(model, word2idx, tag2idx, words, tags)
    return saved_model

def predict(cache_model, input):
    #print("start")
    model, word2idx, tag2idx, words, tags= cache_model.model, cache_model.word2idx, cache_model.tag2idx, cache_model.words, cache_model.tags

    input_x = [[word2idx[w] if w in word2idx else len(words) - 1 for w in
                input.lower().split()]]
    input_x = pad_sequences(maxlen=MODEL_HYPER_PARAMS.MAXLEN, sequences=input_x, padding=MODEL_HYPER_PARAMS.PADDING, value=len(words) - 1)
   # print(input_x[0])
    p = model.predict(np.array([input_x[0]]))
    p = np.argmax(p, axis=-1)
   # print(p[0])
   # print(tag2idx)
    # tag2idx_swap = {v: k for k, v in tag2idx.items()}
    entities = ['time', 'unit']
    time_index = tag2idx.get('time')
    #print(time_index)
    #print(tags)
    dict_entities = {}
    for w, pred in zip(input_x[0], p[0]):
        if (tags[pred] in entities and tags[pred] not in dict_entities):
            #print(words[w - 1], tags[pred])
            dict_entities[tags[pred]] = words[w - 1]

    print(dict_entities)

if __name__ == '__main__':
    base_path = os.path.dirname(__file__)
    print(base_path)
    abs_csv_file_path = os.path.join(base_path, 'input_data.csv')
    abs_model_file_path = os.path.join(base_path, "report_model")
    # train_model_with_input_file(abs_csv_file_path, 500, abs_model_file_path)
    model = load_saved_model(abs_model_file_path)
    input = 'get last 1 month report'
    predict(model, input)
    input = 'give me report for last month'
    predict(model, input)
    input = 'give me feb report'
    predict(model, input)
    input = 'twenty report'
    predict(model, input)
    input = 'can you please generate five month report'
    predict(model, input)
    input = 'hello, pls generate JANUARY report'
    predict(model, input)




"""
data = pd.read_csv("/Users/sakamava/work/tensorflow/django/chatbot/chat_app/bots/report/input_data.csv")
data = data.fillna(method="ffill")
data.head(20)
print("Unique words in corpus:", data['Word'].nunique())
print("Unique tags in corpus:", data['Tag'].nunique())
words = list(set(data["Word"].values))
words.append("ENDPAD")
num_words = len(words)
print(num_words)
tags = list(set(data["Tag"].values))
num_tags = len(tags)
print(num_tags)




getter = SentenceGetter(data)
sentences = getter.sentences
print(sentences)

word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

print(word2idx)
print(tag2idx)

max_len = 50

X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=num_words-1)

y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["0"])
#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.0, random_state=1)

model = tf.keras.Sequential([
       # tf.keras.Input(shape=(max_len)),
       # tf.keras.layers.Embedding(num_words, output_dim=max_len, input_length=max_len),
       # tf.keras.layers.SpatialDropout1D(0.1),
       # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(num_tags, return_sequences=True)),

    #tf.keras.layers.Embedding(num_words, 50, input_length=50),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    #tf.keras.layers.Dropout(0.5),
    #tf.keras.layers.Dense(64, activation='relu'),
    #tf.keras.layers.Dense(num_tags, activation='softmax')

        tf.keras.Input(shape=(max_len)),
        tf.keras.layers.Embedding(num_words, output_dim=max_len, input_length=max_len),
        tf.keras.layers.SpatialDropout1D(0.1),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)), # num_tags or 64 or 32
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(num_tags, activation='softmax')

])
model.summary()

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              #loss = "categorical_crossentropy",
              metrics=["accuracy"])

chkpt = ModelCheckpoint("model_weights.h5", monitor='val_loss',verbose=1, save_best_only=True, save_weights_only=True, mode='min')

early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=1, verbose=0, mode='max', baseline=None, restore_best_weights=False)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        #if (logs.get('accuracy') > 0.98 and logs.get('accuracy') <= 0.99):
        if (logs.get('accuracy') >= 1.0 and logs.get('accuracy') <= 1.0):
            print("\nReached greater than 96.0% and less than 98.0% accuracy so cancelling training!")
            self.model.stop_training = True

#callbacks = [PlotLossesCallback(), chkpt, early_stopping]
callbacks = myCallback() #[ chkpt, early_stopping]
print(X[6])
print(y[6])
history = model.fit(
    x=X, #x_train,
    y=y, #y_train,
    #validation_data=(x_test,y_test),
    batch_size=1,
    epochs=500,
    callbacks=callbacks,
    verbose=1
)

#input = 'give me report for last month'
#input = 'generate last 1 month report'
#input = 'give 1 month report'
#input = 'generate jan report'
#input = 'generate JAN report'
input = 'get last 1 month report'
#input = 'please give me last 1 month report'
#input = 'can you generate report for 1 month'
print(word2idx)
print(len(word2idx))
input_x = [[ word2idx[w] if w in word2idx else num_words-1 for w in input.lower().split()  ]] #[word2idx[w[0]] for w in input]
input_x = pad_sequences(maxlen=max_len, sequences=input_x, padding="post", value=num_words-1)
print(input_x[0])
p = model.predict(np.array([input_x[0]]))
p = np.argmax(p, axis=-1)
print(p[0])
print(tag2idx)
#tag2idx_swap = {v: k for k, v in tag2idx.items()}
entities = ['time','unit']
time_index = tag2idx.get('time')
print(time_index)
print(tags)
dict_entities = {}
for w, pred in zip(input_x[0], p[0]):
    if(tags[pred] in entities and tags[pred] not in dict_entities):
        print(words[w-1], tags[pred])
        dict_entities[tags[pred]] = words[w-1]

print(dict_entities)
"""
