import csv
import os
from typing import NamedTuple, Dict
import json
import jsonpickle
from typing import Optional

from .models import  customIntentSlotDecoder

_bots = ['api', 'test']


class Intent_Output(NamedTuple):
    output: str
    next_intent: Optional[str]
    has_slots: bool = False



_intent_to_output: Dict[str, Dict] = dict()
_intent_to_slots: Dict[str, any] = dict()



def load_intent_to_output(bot_id, file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        _output: Dict[str, Intent_Output] = dict()
        _intent_to_output[bot_id] = _output
        for row in reader:
            _output[row[0]] = Intent_Output(output=row[1], has_slots=row[2] if len(row[2]) > 0 else False, next_intent=row[3] if row[3] is not None and len(row[3])>0 else None)


def load_intents_to_slots(bot_id, file_path):
    try:
        f = open(file_path)
        data = json.load(f, object_hook=customIntentSlotDecoder)
        print(data.apis)
        list_obj = data.apis
        for api in list_obj:
            _intent_to_slots[bot_id+"_"+api.intent] = api
           # print(type(api))
           # print(api.intent)
           # print(len(api.slots))
           # len_slot = len(api.slots)
           # if(len_slot>0):
           #   print(f'slot 0 - {len_slot} {api.slots[0]} {api.intent}')

        #for intents in _intent_to_slots:
        #    print(f'{intents} {_intent_to_slots[intents]}')
    except FileNotFoundError:
        print(f'file not found {file_path}')

def get_intent_details(bot_id,intent):
    if bot_id+"_"+intent in _intent_to_slots:
        return _intent_to_slots[bot_id+"_"+intent]

def get_output(bot_id, prediction):
    print(_intent_to_output)
    output_pred = _intent_to_output[bot_id]
    print(output_pred)
    print(f'{output_pred[prediction].output} , {output_pred[prediction].has_slots}')
    return output_pred[prediction].output

def get_next_intent_data(bot_id, prediction):
    print(_intent_to_output)
    output_pred = _intent_to_output[bot_id]
    print(f'{output_pred[prediction].output} , {output_pred[prediction].has_slots},  {output_pred[prediction].next_intent}')
    if output_pred[prediction].next_intent is not None:
         return output_pred[output_pred[prediction].next_intent].output


for bot_id in _bots:
    base_path = os.path.join(os.path.dirname(__file__), 'bots', bot_id)
    file_path = os.path.join(base_path, 'intent_to_output.csv')
    load_intent_to_output(bot_id, file_path)
    intents_slots = os.path.join(base_path, 'intents_slots.json')
    load_intents_to_slots(bot_id, intents_slots)

