import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras.engine
import io
import os
import json
from dataclasses import dataclass
import re


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
        print(logs.get('accuracy'))
        if logs.get('accuracy') >= 0.9999: #and logs.get('accuracy') <= 1.0):
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
      #  callbacks=callbacks,
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

def preprocess(sentence):
    sentence = sentence.lower()
    characters_to_remove = ",-/?."
    regex_pattern = "[" + re.escape(characters_to_remove) + "]"
    sentence = re.sub(regex_pattern, " ", sentence)
    sentence = sentence.replace("'s", "")
    sentence = sentence.replace("hours","hr")
    sentence = sentence.replace("hrs", "hr")
   # print(sentence)
    return sentence

def predict(cache_model, input,entities):
    #print("start")
    model, word2idx, tag2idx, words, tags= cache_model.model, cache_model.word2idx, cache_model.tag2idx, cache_model.words, cache_model.tags

    input_x = [[word2idx[w] if w in word2idx else len(words) - 1 for w in
                preprocess(input).split()]]
                #input.lower().split()]]
    input_x = pad_sequences(maxlen=MODEL_HYPER_PARAMS.MAXLEN, sequences=input_x, padding=MODEL_HYPER_PARAMS.PADDING, value=len(words) - 1)
   # print(input_x[0])
    p = model.predict(np.array([input_x[0]]))
    p = np.argmax(p, axis=-1)
   # print(p[0])
   # print(tag2idx)
    # tag2idx_swap = {v: k for k, v in tag2idx.items()}
    #entities = ['time', 'unit']
    #time_index = tag2idx.get('time')
    #print(time_index)
   # print(tags)
   # print(p[0])
   # print(input_x[0])
    dict_entities = {}
    for w, pred in zip(input_x[0], p[0]):
        if (tags[pred] in entities):
             if tags[pred] not in dict_entities:
                dict_entities[tags[pred]] = [words[w - 1]]
             else:
                dict_entities[tags[pred]].append(words[w - 1])

    print(input,' ',dict_entities)
    return dict_entities

if __name__ == '__main__1':
    print(preprocess("Today'S yesterday's jan,2nd   2022 2022-01-22 04/4/2023"))
    base_path = os.path.dirname(__file__)
    print(base_path)
    abs_csv_file_path = os.path.join(base_path, 'input_data.csv')
    abs_model_file_path = os.path.join(base_path, "report_model")
    # train_model_with_input_file(abs_csv_file_path, 10, abs_model_file_path)
    model = load_saved_model(abs_model_file_path)
    entities = ['state', 'time', 'unit', 'month', 'year']
    input = "today's report"
    predict(model, input, entities)

if __name__ == '__main__':
    base_path = os.path.dirname(__file__)
    print(base_path)
    abs_csv_file_path = os.path.join(base_path, 'input_data.csv')
    abs_model_file_path = os.path.join(base_path, "report_model")
    #train_model_with_input_file(abs_csv_file_path, 10, abs_model_file_path)
    model = load_saved_model(abs_model_file_path)
    entities = ['state', 'time', 'unit', 'month', 'year']
    input = 'get last 1 month report'
    predict(model, input, entities)
    input = 'give me report for last month'
    predict(model, input, entities)
    input = 'give me feb report'
    predict(model, input, entities)
    input = 'twenty report'
    predict(model, input, entities)
    input = 'can you please generate five month report'
    predict(model, input, entities)
    input = 'hello, pls generate JANUARY report'
    predict(model, input, entities)
    input = 'i, pls generate report'
    predict(model, input, entities)
    input = 'generate report for 2 jan 2021'
    predict(model, input, entities)
    input = 'give me report for 21 February 2021'
    predict(model, input, entities)
    input = 'give me report betwen 21 February 2021 to 22 apr 2021'
    predict(model, input, entities)
    input = 'give me current month report'
    predict(model, input, entities)
    input = 'give me last yr report'
    predict(model, input, entities)
    input = 'generate report between 1 jan 2022 and 29 jun 2022 report'
    predict(model, input, entities)
    input = 'generate report from 1 jan 2022 to 29 jun 2022 report'
    predict(model, input, entities)
    input = 'generate report from 1 12 2022 to 01 06 2023 report'
    predict(model, input, entities)
    input = 'generate report from 1 12 2022 to 1 6 2023 report'
    predict(model, input, entities)
    input = 'generate report from 2022 06 12 to 2022 07 13 report'
    predict(model, input, entities)
    input = 'generate report from 2022 08 12 to 2023 05 21 report'
    predict(model, input, entities)
    input = 'generate report from march 2022 to june 2022 report'
    predict(model, input, entities)
    input = 'generate report from july 2022 to november 2022 report'
    predict(model, input, entities)
    input = 'generate report from sep 2022 to april 2023 report'
    predict(model, input, entities)
    input = 'generate report from may 2022 to aug 2023 report'
    predict(model, input, entities)
    input = 'generate report from oct 2022 to nov 2023 report'
    predict(model, input, entities)
    input = 'generate report from 2nd sep 2022 to 4th nov 2023 report'
    predict(model, input, entities)
    input = 'generate report from 8th june 2022 to 1st july 2023 report'
    predict(model, input, entities)
    input = 'generate report from 3rd may 2022 to 5th dec 2023 report'
    predict(model, input, entities)
    input = 'report from january 2021 to september 2022 report'
    predict(model, input, entities)
    input = 'report from jul to december'
    predict(model, input, entities)
    input = 'report from nov 2020 to august 2020'
    predict(model, input, entities)
    input = 'give me report from 2021 02 22 to 2023 08 30 report'
    predict(model, input, entities)
    input = 'give me report from 2023 04 22 to 2023 9 20 report'
    predict(model, input, entities)
    input = 'generate report from 12 17 2023 to 03 04 2023 report'
    predict(model, input, entities)
    input = 'generate report from 1 8 2023 to 03 10 2023 report'
    predict(model, input, entities)
    input = 'today report'
    predict(model, input, entities)
    input = "today's report"
    predict(model, input, entities)
    input = 'yesterday report'
    predict(model, input, entities)
    input = "report for today's"
    predict(model, input, entities)
    input = "report for yesterday's"
    predict(model, input, entities)
    input = 'generate report from 10-9-2023 to 08-7-2023'
    predict(model, input, entities)
    input = 'generate report from 12-27-2022 to 11-17-2023 report'
    predict(model, input, entities)
    input = 'generate report from 9/5/2022 to 8/7/2023'
    predict(model, input, entities)
    input = 'generate report from 6/15/2021 to 04/07/2022 report'
    predict(model, input, entities)
    input = 'generate report from 03/05/2022 to 02/19/2023'
    predict(model, input, entities)
    input = 'generate report from 01/01/2023 to 01/31/2023 report'
    predict(model, input, entities)
    input = 'generate report from 2023/02/01 to 2023/02/28'
    predict(model, input, entities)
    input = 'report for last 4 hours'
    predict(model, input, entities)
    input = 'report for last 4 hour'
    predict(model, input, entities)
    input = 'report for last 8 hrs'
    predict(model, input, entities)
    input = 'last 8 years'
    predict(model, input, entities)


# remove ,'-/ with spaces
# remove extra space
# train with todays yesterdays data
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
