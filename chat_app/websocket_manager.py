from .services import process_bot_input
from .rest_models import PredictArgs


def process_prediction(data):
    obj = PredictArgs(**data)
    data = process_bot_input(obj)
    return data

