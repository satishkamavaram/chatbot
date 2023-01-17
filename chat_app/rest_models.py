from django.db import models
from pydantic import BaseModel, Extra,Field
from typing import Optional

# Create your models here.
class ModelArgs(BaseModel, extra=Extra.forbid):
    key: Optional[str] = Field(exclude=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class StatusArgs(BaseModel):
    statusCode: str = "S0000"
    statusMessage: str = "Success"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BotArgs(StatusArgs):
    botId: Optional[str] = None

    def __str__(self):
        return self.botId

class PredictArgs(BotArgs):
    input: Optional[str] = None
    intent: Optional[str] = None
    output: Optional[str] = None
    sessionId: Optional[str] = None


    def __str__(self):
        return self.input

class Message(BaseModel):
    content: str = None
    content_type: str = "plain/text"
    defautValue: Optional[str] = None
    listOfPossibleValues: Optional[list] = None


class BotResponseArgs(BotArgs):
    input: Optional[str] = None
    intent: Optional[str] = None
    messages: list[Message] = None
    sessionId: Optional[str] = None

    def __str__(self):
        return self.input

   # class Config:
   #     arbitrary_types_allowed = True


