from django.urls import path

from . import views

app_name = 'chat_app'
urlpatterns = [
    path('', views.index, name='index'),
    path('class', views.ChatView.as_view(), name='class-my-view'),
    path('bot', views.ChatBotView.as_view(), name='bot-view'),
]