# Exposes REST API for training the chatbot model and do the prediction using Tensorflow

## REST APIs
* For training model: http://localhost:8000/chat/bot GET 
* For prediction: http://localhost:8000/chat/bot POST {"botId":"api","input":"hi","sessionId": "In-FlKiIQ5kx-CsecR3JWAs"}

## WebSocket
* Websocket: 
  * endpoint: ws://localhost:8000/ws/socket-server/ 
  * json input body: {"botId":"api","input":"hi","sessionId": "In-FlKiIQ5kx-CsecR3JWAs"}
  * 
## To run the server
python3 manage.py runserver 

## Supported botIds
  * api
  * report
