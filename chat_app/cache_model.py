from .com.api.chat.model.lstm_model import *
from pathlib import Path
import sys
import re

_cached_models = {}


def train_model_cache(bot_id, abs_csv_file_path, epochs, abs_model_file_path):
    print("training")
    model = train_model_with_input_file(abs_csv_file_path, epochs, abs_model_file_path)
    _cached_models[bot_id] = model
    print(_cached_models)


def predict_model_cache(bot_id, save_model_path, input_sentence):
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
    return predict(model, input_sentence)


def validate(validateFunc, arg):
    print(f'args: {validateFunc} {arg}')
    getattr(sys.modules[__name__], validateFunc)(arg)


def validate_slot(slot, input):
    print(f'args: {slot} {input}')
    if slot.regexValidator is not None:
        print(f'validating regex {slot.regexValidator} {input}')
        return regex_validator(slot.regexValidator, input)
    if slot.validationFunction is not None:
        print(f'validating function {slot.validationFunction} {input}')
        getattr(sys.modules[__name__], slot.validationFunction)(input)
    if slot.listOfSupportedValues is not None and len(slot.listOfSupportedValues)>0:
        print(f'validating list of values {slot.listOfSupportedValues} {input}')
        return list_validator(slot.listOfSupportedValues, input)



def regex_validator(regex, input):
    if not re.match(regex, input):
        raise ValueError(f'{input} is invalid')


def list_validator(list_of_values, input):
    if input in list_of_values:
        return True
    else:
        raise ValueError(f'{input} is invalid')


def validate_filename(filename: str) -> str:
    print(f'filename: {filename}')
    file_path = Path(filename)
    print(file_path)
    print(file_path.parent.exists())
    if file_path.parent.exists():
        raise ValueError(f'Directory for "{filename}" does not exist. Please enter valid {filename}:')
    return filename


index = 0


def validate_filter(filename: str) -> str:
    print(f'validate_filter: {filename}')
    global index
    if index == 0:
        index = index + 1
        raise ValueError(f'Directory for "{filename}" does not exist. Please enter valid {filename}:')
    return filename


def validate_limit(filename: str) -> str:
    print(f'validate_limit: {filename}')
    return filename


def validate_page(filename: str) -> str:
    print(f'validate_page: {filename}')
    return filename
