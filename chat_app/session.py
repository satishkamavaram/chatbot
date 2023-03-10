from pydantic import BaseModel, Extra, Field
from typing import Optional, Dict, List
from enum import Enum
import secrets
from .models import Slot


class Intent_State(Enum):
    IN_PROGRESS = 0
    COMPLETED = 1


class Type(Enum):
    INTENT = 'intent'
    SLOT = 'slot'


class Rest_Api:
    path_params: Dict[str, str]
    query_params: Dict[str, str]
    headers: Dict[str, str]
    payload: str
    uri: str

    def __init__(self, uri, path_params: Dict = {}, query_params: Dict = {}, headers: Dict = {}, payload=None):
        self.uri = uri
        self.path_params = path_params
        self.query_params = query_params
        self.headers = headers
        self.payload = payload


class CurrentSlot:
    slotNo: int  # intent or slot
    slotName: str  # name of the slot

    def __init__(self, slotNo, slotName):
        self.slotNo = slotNo
        self.slotName = slotName


class Intent:
    name: str
    state: Intent_State  # enum
    current_slot: CurrentSlot  # current slot number , slot name
    slot: any
    total_slots: int

    def __init__(self, name, state, current_slot: CurrentSlot, slot,total_slots):
        self.name = name
        self.state = state
        self.current_slot = current_slot
        self.slot = slot
        self.total_slots = total_slots


#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)




class DialogAction:
    type: Type  # intent or slot
    slot_to_elicit: str  # name of the slot

    def __init__(self, type, slot_to_elicit):
        self.type = type
        self.slot_to_elicit = slot_to_elicit

#   def __init__(self, **kwargs):
#       super().__init__(**kwargs)


class Session:
    session_id: str
    intent: Intent
    dialog_action: DialogAction
    rest_api: Rest_Api

    def __init__(self, session_id, intent, dialog_action, rest_api: Rest_Api):
        self.self = self
        self.session_id = session_id
        self.intent = intent
        self.dialog_action = dialog_action
        self.rest_api = rest_api

# def __init__(self, **kwargs):
#     super().__init__(**kwargs)


_session_cache: Dict[str, Session] = dict()


def get_session(session_id):
    return _session_cache.get(session_id)


def get_unique_session_id():
    return secrets.token_urlsafe(16)


def save_session(session_id: str, session: Session):
    _session_cache[session_id] = session

def remove_session(session_id: str):
    _session_cache.pop(session_id, None)
