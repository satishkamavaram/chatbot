from django.db import models
from pydantic import BaseModel, Extra,Field
from typing import Optional
from collections import namedtuple


class Slot:
    def __init__(self, paramName: str, question: str, mandatory: bool, dataType: str, type: str,
                 validationFunction: Optional[str] = None, defaultValue: Optional[any] = None,
                 listOfSupportedValues:Optional[list] = [], regexValidator: Optional[str] = None,) -> None:
        self.paramName = paramName
        self.question = question
        self.validationFunction = validationFunction
        self.regexValidator = regexValidator
        self.listOfSupportedValues = listOfSupportedValues
        self.mandatory = mandatory
        self.defaultValue = defaultValue
        self.dataType = dataType
        self.type = type


class Api:
    def __init__(self, intent: str, apiName: str, apiDesc: str, httpMethod: str, uri: str,
                 slots: Optional[list[Slot]] = []) -> None:
        self.intent = intent
        self.apiName = apiName
        self.apiDesc = apiDesc
        self.httpMethod = httpMethod
        self.uri = uri
        self.slots = slots


class Apis:
    def __init__(self, apis: list[Api]) -> None:
        self.apis = apis


def customIntentSlotDecoder(apisDict):
    return namedtuple('Intents_Slots', apisDict.keys())(*apisDict.values())

