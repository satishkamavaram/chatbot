from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from .rest_models import PredictArgs, BotArgs, BotResponseArgs, Message

from .com.api.chat.model.lstm_model import *
from .util import get_output, get_intent_details, get_next_intent_data

from .cache_model import predict_model_cache, validate_slot
from .session import *


def process_bot_input(obj: PredictArgs):
    base_path = os.path.join(os.path.dirname(__file__), 'bots', obj.botId)
    abs_model_file_path = os.path.join(base_path, "model")
    if obj.sessionId is not None and get_session(obj.sessionId) is not None:
        session = get_session(obj.sessionId)
        if session.intent.state == Intent_State.IN_PROGRESS:
            try:
                validate_slot(session.intent.slot, obj.input)
            except ValueError as ve:
                message = get_message(str(ve), "plain/text", session)
                return get_http_response(obj.input, session.intent.name, obj.sessionId, [message])

            if session.intent.current_slot.slotNo < session.intent.total_slots - 1:
                current_slot = session.intent.current_slot.slotNo + +1
                api = get_intent_details(obj.botId, session.intent.name)
                session.intent.current_slot = CurrentSlot(slotNo=current_slot,
                                                          slotName=api.slots[current_slot].paramName)
                session.intent.slot = api.slots[current_slot]
                session.dialog_action.slot_to_elicit = api.slots[current_slot].paramName
                message = get_message(api.slots[current_slot].question, "plain/text", session)
                return get_http_response(obj.input, session.intent.name, obj.sessionId, [message])
            else:
                ## execute the api
                session.intent.state = Intent_State.COMPLETED
                return get_http_response(obj.input, session.intent.name, obj.sessionId,
                                         [get_message('We are processing your request', 'plain/text'),
                                          get_message(get_next_intent_data(obj.botId, session.intent.name),
                                                      'plain/text')])

    prediction = predict_model_cache(obj.botId, abs_model_file_path, obj.input)
    api = get_intent_details(obj.botId, prediction)
    if api is None:
        list_of_messages = [get_message(get_output(obj.botId, prediction), 'plain/text')]
        return get_http_response(obj.input, prediction, None, list_of_messages)
    elif len(api.slots) > 0:
        session_id = get_session_id(obj)
        session = create_and_save_session_with_slots(session_id, prediction, api)
        list_of_messages = [get_message(api.slots[0].question, "plain/text", session)]
        return get_http_response(obj.input, prediction, session_id, list_of_messages)
    else:
        session_id = get_session_id(obj)
        create_and_save_session_without_slots(session_id, prediction)
        list_of_messages = [get_message(get_output(obj.botId, prediction), 'plain/text'),
                            get_message(get_next_intent_data(obj.botId, prediction), 'plain/text')]
        return get_http_response(obj.input, prediction, session_id, list_of_messages)


def get_http_response(input, intent, session_id, list_of_messages):
    obj = BotResponseArgs(input=input, intent=intent, messages=list_of_messages, sessionId=session_id)
    data = obj.json(exclude_none=True)
    return HttpResponse(data, content_type='application/json')


def get_message(content:str, content_type:str, session:Session=None):
    if session is None:
        return Message(content=content, content_type=content_type)
    else:
        return Message(content=content, content_type=content_type,
                       defautValue=session.intent.slot.defaultValue if session.intent.slot.defaultValue is not None else None,
                       listOfPossibleValues=session.intent.slot.listOfSupportedValues if session.intent.slot.listOfSupportedValues is not None and len(
                           session.intent.slot.listOfSupportedValues) > 0 else None)


def get_session_id(request_obj):
    return request_obj.sessionId if request_obj.sessionId is not None and get_session(
        request_obj.sessionId) is not None else get_unique_session_id()


def create_and_save_session_with_slots(session_id:str, prediction:str, api):
    current_slot = CurrentSlot(slotNo=0, slotName=api.slots[0].paramName)
    print(api.slots)
    intent = Intent(name=prediction, state=Intent_State.IN_PROGRESS, current_slot=current_slot, slot=api.slots[0],
                    total_slots=len(api.slots))
    dialog_action = DialogAction(type=Type.SLOT, slot_to_elicit=api.slots[0].paramName)
    session = Session(session_id=session_id, intent=intent, dialog_action=dialog_action)
    save_session(session_id, session)
    return session


def create_and_save_session_without_slots(session_id:str, prediction:str):
    intent = Intent(name=prediction, state=Intent_State.COMPLETED, current_slot=None, slot=None,
                    total_slots=0)
    dialog_action = DialogAction(type=Type.INTENT, slot_to_elicit=None)
    session = Session(session_id=session_id, intent=intent, dialog_action=dialog_action)
    save_session(session_id, session)
    return session
