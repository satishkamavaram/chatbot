import json
from channels.generic.websocket import WebsocketConsumer
from .session import *
# documentation : https://channels.readthedocs.io/en/stable/topics/consumers.html

_ws_sessions = set()

class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        print(self)
        self.scope["sessionId"] = get_unique_session_id()
        print(f'establishing connection {self.scope["session"]}')
        print(self.scope)
        self.send(text_data = json.dumps({
            'type' : 'connection_established',
            'message' : 'Hello , we are connected'
        }))

    def receive(self, text_data):
        print(f'receive: {text_data}')
        print(self)
        print(f'scope {self.scope}')

        text_data_json = json.loads(text_data)
        print(f'text_data_json {text_data_json}')
        message = text_data_json['message']
        print(f'message {message}')
        for i in range(5):
            print(f'iteration {i}')
            self.send(text_data=json.dumps({
                'type': 'connection_established',
                'message': message
            }))
            print(f'end iteration {i}')

    def disconnect(self,code):
        print("hi...disconnecting", code, self, self.scope["sessionId"])