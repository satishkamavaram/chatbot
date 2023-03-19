from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from .rest_models import PredictArgs, BotArgs, BotResponseArgs, Message

from .com.api.chat.model.lstm_model import *
from .util import get_output, get_intent_details, get_next_intent_data

from .cache_model_entity import predict_model_cache
from .session import *
from .rest_api import execute_post_bpa_api


def process_report_bot_input(obj: PredictArgs):
    base_path = os.path.join(os.path.dirname(__file__), 'bots', obj.botId)
    abs_model_file_path = os.path.join(base_path, "model")
    entities = ['state', 'time', 'unit', 'month', 'year']
    start_end_date = predict_model_cache(obj.botId, abs_model_file_path, obj.input,entities)
    prediction = 'processinstances'
    api = get_intent_details(obj.botId, prediction)
    print('api', api)
    print('api.requestBody', api.requestBody)
    json_body = api.requestBody.format(**start_end_date)
    print('json_body', json_body)
    json_response = execute_post_bpa_api(api.uri, json_body, None)
    print('json_response', json_response)
    json_response = summarize_response(json_response)
    session_id = get_session_id(obj)
    list_of_messages = [get_message(json_response, 'application/json')]
    return get_http_response(obj.input, prediction, session_id, list_of_messages,start_end_date)


def summarize_response(json_response: dict):
    state_counts = {}
    if 'statusCode' not in json_response:
        for item in json_response:
            state = item['state']
            if state not in state_counts:
                state_counts[state] = 1
            else:
                state_counts[state] += 1
        return state_counts
    return json_response


def get_http_response(input, intent, session_id, list_of_messages,start_end_date):
    obj = BotResponseArgs(input=input, intent=intent, messages=list_of_messages, sessionId=session_id, additionalInfo=start_end_date)
    data = obj.json(exclude_none=True)
    return data
    #return HttpResponse(data, content_type='application/json')


def get_message(content: str, content_type: str, session: Session = None):
    return Message(content=content, content_type=content_type)


def get_session_id(request_obj):
    return request_obj.sessionId if request_obj.sessionId is not None else get_unique_session_id()


