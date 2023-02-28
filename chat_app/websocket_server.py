import json
from channels.generic.websocket import WebsocketConsumer
from .session import *
from .websocket_manager import *
# documentation : https://channels.readthedocs.io/en/stable/topics/consumers.html

_ws_sessions = set()

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        print(f'connect ws sessions: {_ws_sessions}')
        self.accept()
        print(self)
        self.scope["sessionId"] = get_unique_session_id()
        print(f'establishing connection {self.scope["session"]}')
        print(self.scope)
        _ws_sessions.add(self.scope["sessionId"])
        save_session(self.scope["sessionId"], None)
     #   self.send(text_data = json.dumps({
     #       'type' : 'connection_established',
     #       'message' : 'Hello , we are connected'
     #   }))

    def receive(self, text_data):
        print(f'receive: {text_data}')
        if text_data == '1':
            #self.send(text_data='1')
            pass
        else:
            print(self)
            print(f'scope {self.scope}')
            text_data_json = json.loads(text_data)
            text_data_json['sessionId'] = self.scope["sessionId"]
            print(f'text_data_json {text_data_json}')
            input = text_data_json['input']
            print(f'input {input}')
            response = process_prediction(text_data_json)
            print(response)
            self.send(text_data=response)


    def disconnect(self,code):
        print("hi...disconnecting", code, self, self.scope["sessionId"])
        _ws_sessions.discard(self.scope["sessionId"])
        remove_session(self.scope["sessionId"])
        print(f'ws sessions: {_ws_sessions}')
        self.close()