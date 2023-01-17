from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from rest_framework.response import Response
from django.views import View
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.core import serializers
import json
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer, TemplateHTMLRenderer
from rest_framework.decorators import action
from rest_framework import status, viewsets
from .rest_models import PredictArgs, BotArgs

from .com.api.chat.model.lstm_model import *
from .util import get_output

from .cache_model import train_model_cache, predict_model_cache, validate
from .services import process_bot_input


@csrf_exempt
def index(request):
    return HttpResponse("Hello, world. ")

@method_decorator(csrf_exempt, name='dispatch')
#@renderer_classes((TemplateHTMLRenderer, JSONRenderer))
class ChatView(APIView):

    def get(self, request, *args, **kwargs):
        obj = PredictArgs(key1='key',input='sfd',intent='sd')
       # predict = obj.dict()
        #data = json.dumps(predict)
       # data = obj.json(exclude={'key1'})
        data = obj.json()
        return HttpResponse(data, content_type='application/json')

    def post(self, request, *args, **kwargs):
        print(request.headers)
        print(request.query_params)
        print(request.data)
        obj = PredictArgs(**request.data)
        predict = obj.dict()
        print(predict)
        data = json.dumps(predict)
        print(data)
        return HttpResponse(data, content_type='application/json')

@method_decorator(csrf_exempt, name='dispatch')
class ChatBotView(viewsets.ModelViewSet):
    @action(detail=False)
    def get(self, request, *args, **kwargs):
        base_path = os.path.dirname(__file__)
        abs_csv_file_path = os.path.join(base_path, 'intent_to_input.csv')
        abs_model_file_path = os.path.join(base_path, "my_model")
        train_model_with_input_file(abs_csv_file_path, 120, abs_model_file_path)
        obj = PredictArgs(statusMessage='Training Successfully Completed')
        data = obj.json(exclude_none=True)
        return HttpResponse(data, content_type='application/json')

    @action(methods=['post'], detail=False, url_path='train')
    def train_bot(self, request, *args, **kwargs):
        obj = BotArgs(**request.data)
        base_path = os.path.join(os.path.dirname(__file__), 'bots', obj.botId)
        abs_csv_file_path = os.path.join(base_path,  'intent_to_input.csv')
        abs_model_file_path = os.path.join(base_path, "model")
      #  train_model_with_input_file(abs_csv_file_path, 120, abs_model_file_path)
        train_model_cache(obj.botId, abs_csv_file_path, 1000, abs_model_file_path)
        obj = PredictArgs(statusMessage='Training Successfully Completed')
        data = obj.json(exclude_none=True)
        return HttpResponse(data, content_type='application/json')

    @action(methods=['post'], detail=False, url_path='')
    def post(self, request, *args, **kwargs):
        obj = PredictArgs(**request.data)
        return process_bot_input(obj)

"""   
        base_path = os.path.join(os.path.dirname(__file__), 'bots', obj.botId)
        abs_model_file_path = os.path.join(base_path, "model")
        #prediction = predict(abs_model_file_path, obj.input)
        prediction = predict_model_cache(obj.botId,abs_model_file_path, obj.input)
        obj = PredictArgs(input=obj.input, intent=prediction, output=get_output(obj.botId, prediction))
        data = obj.json(exclude_none=True)
        fields = {'name': 'validate_filename'}
        validate('validate_filename', 'testsfasd')
        return HttpResponse(data, content_type='application/json')
"""