# Exposes REST API for training the chatbot model and do the prediction using Tensorflow

## APIs
* For training model: http://localhost:8000/chat/bot GET 
* For prediction: http://localhost:8000/chat/bot POST { "input": "getdevices" }

## To run the server
python3 manage.py runserver 

