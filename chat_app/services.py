from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from .rest_models import PredictArgs, BotArgs, BotResponseArgs, Message

from .com.api.chat.model.lstm_model import *
from .util import get_output, get_intent_details, get_next_intent_data

from .cache_model import predict_model_cache, validate_slot
from .session import *
from .rest_api import execute_bpa_api


def process_bot_input(obj: PredictArgs):
    base_path = os.path.join(os.path.dirname(__file__), 'bots', obj.botId)
    abs_model_file_path = os.path.join(base_path, "model")
    if obj.sessionId is not None: # and get_session(obj.sessionId) is not None:
        session = get_session(obj.sessionId)
        if session is not None and session.intent.state == Intent_State.IN_PROGRESS:
            try:
                if not (obj.input is not None and len(obj.input.strip()) == 0 and session.intent.slot.mandatory is False):
                    validate_slot(session.intent.slot, obj.input) # if mandatory is false, then ignore empty value
            except ValueError as ve:
                message = get_message(str(ve)+','+session.intent.slot.question, "plain/text", session)
                return get_http_response(obj.input, session.intent.name, obj.sessionId, [message])

            if session.intent.slot.type == "query_param":
                add_query_param_to_session(session, session.intent.slot.paramName, obj.input)

            if session.intent.slot.type == "path_param":
                add_path_param_to_session(session, session.intent.slot.paramName, obj.input)

              # if more than one slot , goes insdie this loop
            if session.intent.current_slot.slotNo < session.intent.total_slots - 1:
                current_slot = session.intent.current_slot.slotNo + +1
                api = get_intent_details(obj.botId, session.intent.name)
                session.intent.current_slot = CurrentSlot(slotNo=current_slot,
                                                          slotName=api.slots[current_slot].paramName)
                session.intent.slot = api.slots[current_slot]
                session.dialog_action.slot_to_elicit = api.slots[current_slot].paramName
                message = get_message(api.slots[current_slot].question, "plain/text", session)
                return get_http_response(obj.input, session.intent.name, obj.sessionId, [message])
            else: # all slots are answered by user for an intent, going to execute api
                ## execute the api
                json_response = execute_bpa_api(session.rest_api.uri, session.rest_api.path_params, session.rest_api.query_params)
                session.intent.state = Intent_State.COMPLETED
                return get_http_response(obj.input, session.intent.name, obj.sessionId,
                                         [get_message('We are processing your request', 'plain/text')
                                          if json_response is None else
                                          get_message(json_response, 'application/json'),
                                          get_message(get_next_intent_data(obj.botId, session.intent.name),
                                                      'plain/text')])

    prediction = predict_model_cache(obj.botId, abs_model_file_path, obj.input)
    api = get_intent_details(obj.botId, prediction)
    if api is None: # if bot is not an API , goes inside this block
        session_id = get_session_id(obj)
        create_and_save_session_without_slots(session_id, prediction, api)
        list_of_messages = [get_message(get_output(obj.botId, prediction), 'plain/text'),
                            get_message(get_next_intent_data(obj.botId, prediction), 'plain/text')]
        return get_http_response(obj.input, prediction, session_id, list_of_messages)
    elif len(api.slots) > 0: # if slots are available for an intent, goes inside this block
        session_id = get_session_id(obj)
        session = create_and_save_session_with_slots(session_id, prediction, api)
        list_of_messages = [get_message(api.slots[0].question, "plain/text", session)]
        return get_http_response(obj.input, prediction, session_id, list_of_messages)
    else: # if no slots are available for an intent , goes inside this block
        ## execute the api
        json_response = execute_bpa_api(api.uri, None, None)
        session_id = get_session_id(obj)
        create_and_save_session_without_slots(session_id, prediction, api)
        list_of_messages = [get_message(get_output(obj.botId, prediction), 'plain/text') if json_response is None else
                            get_message(json_response, 'application/json'),
                            get_message(get_next_intent_data(obj.botId, prediction), 'plain/text')]
        return get_http_response(obj.input, prediction, session_id, list_of_messages)


def get_http_response(input, intent, session_id, list_of_messages):
    obj = BotResponseArgs(input=input, intent=intent, messages=list_of_messages, sessionId=session_id)
    data = obj.json(exclude_none=True)
    return data
    #return HttpResponse(data, content_type='application/json')


def get_message(content: str, content_type: str, session: Session = None):
    if session is None:
        return Message(content= content if content is not None and len(content) > 0 else "We are processing your request", content_type=content_type)
    else:
        return Message(content=content, content_type=content_type,
                       defautValue=session.intent.slot.defaultValue if session.intent.slot.defaultValue is not None else None,
                       mandatory=session.intent.slot.mandatory if session.intent.slot.mandatory is not None else None,
                       listOfPossibleValues=session.intent.slot.listOfSupportedValues if session.intent.slot.listOfSupportedValues is not None and len(
                           session.intent.slot.listOfSupportedValues) > 0 else None)


def get_session_id(request_obj):
    #return request_obj.sessionId if request_obj.sessionId is not None and get_session(
    #    request_obj.sessionId) is not None else get_unique_session_id()
    return request_obj.sessionId if request_obj.sessionId is not None else get_unique_session_id()


def create_and_save_session_with_slots(session_id: str, prediction: str, api):
    current_slot = CurrentSlot(slotNo=0, slotName=api.slots[0].paramName)
    print(api.slots)
    intent = Intent(name=prediction, state=Intent_State.IN_PROGRESS, current_slot=current_slot, slot=api.slots[0],
                    total_slots=len(api.slots))
    dialog_action = DialogAction(type=Type.SLOT, slot_to_elicit=api.slots[0].paramName)
    session = Session(session_id=session_id, intent=intent, dialog_action=dialog_action, rest_api=Rest_Api(api.uri))
    save_session(session_id, session)
    return session


def create_and_save_session_without_slots(session_id: str, prediction: str, api):
    intent = Intent(name=prediction, state=Intent_State.COMPLETED, current_slot=None, slot=None,
                    total_slots=0)
    dialog_action = DialogAction(type=Type.INTENT, slot_to_elicit=None)
    session = Session(session_id=session_id, intent=intent, dialog_action=dialog_action, rest_api=Rest_Api(api.uri if api is not None else None))
    save_session(session_id, session)
    return session


def add_path_param_to_session(session: Session, key: str, value: str):
    session.rest_api.path_params[key] = value


def add_query_param_to_session(session: Session, key: str, value: str):
    session.rest_api.query_params[key] = value
