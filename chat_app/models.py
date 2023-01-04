from django.db import models
from pydantic import BaseModel, Extra,Field
from typing import Optional

# Create your models here.
class ModelArgs(BaseModel, extra=Extra.forbid):
    key1: Optional[str] = Field(exclude=True)
    statusCode: str = "S0000"
    statusMessage: str = "Success"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class PredictArgs(ModelArgs):
    input: Optional[str] = None
    intent: Optional[str] = None
    output: Optional[str] = None

    def __str__(self):
        return self.input

