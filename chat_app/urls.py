from django.urls import path

from . import views
from rest_framework import routers

app_name = 'chat_app'
urlpatterns = [
    path('', views.index, name='index'),
    path('class', views.ChatView.as_view(), name='class-my-view'),
    path('bot/train', views.ChatBotView.as_view({'post': 'train_bot'}), name='bot-view'),
    path('bot', views.ChatBotView.as_view({'post': 'post', 'get': 'get'}), name='bot-view'),
]
