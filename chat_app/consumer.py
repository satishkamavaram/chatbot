import json
from channels.generic.websocket import WebsocketConsumer


class ChatConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()
        print(self.scope["session"])
        self.send(text_data = json.dumps({
            'type' : 'connection_established',
            'message' : 'Hello , we are connected'
        }))

    def receive(self, text_data):
        print(text_data)
        print(self)
        print(self.scope)

        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        self.send(text_data=json.dumps({
            'type': 'connection_established',
            'message': message
        }))

    def disconnect(self,code):
        print("hi",code)