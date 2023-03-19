from .com.api.chat.model.entity_model import *
from .date_time import *
from pathlib import Path
import sys
import re

_cached_models = {}


def train_model_cache(bot_id, abs_csv_file_path, epochs, abs_model_file_path):
    print("training")
    model = train_model_with_input_file(abs_csv_file_path, epochs, abs_model_file_path)
    _cached_models[bot_id] = model
    print(_cached_models)


def predict_model_cache(bot_id, save_model_path, input_sentence, entities: list):
    print(f'predict {bot_id} {save_model_path} {input_sentence}')
    if bot_id not in _cached_models:
        print("1")
        model = _cached_models[bot_id] = load_saved_model(save_model_path)
        print(model)
        print(_cached_models)
    else:
        print("2")
        model = _cached_models[bot_id]
    print(_cached_models)
    data = predict(model, input_sentence, entities)
    print(data)
    start_date, end_date  = get_start_end_date(data)
    start_end_date_json = {'start_date': start_date, 'end_date': end_date}
    print("start_end_date_json", start_end_date_json)
    return start_end_date_json


