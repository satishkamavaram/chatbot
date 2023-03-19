from .services import process_bot_input
from .report_services import process_report_bot_input
from .rest_models import PredictArgs


def process_prediction(data):
    obj = PredictArgs(**data)
    if obj.botId == 'report':
        data = process_report_bot_input(obj)
    else:
        data = process_bot_input(obj)
   # data = process_bot_input(obj)
    return data
