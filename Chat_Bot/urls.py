# Chat_Bot/urls.py

from django.urls import path
from .views import store_groq_key, upload_pdf_and_create_bot, get_bot_response

urlpatterns = [
    path('store-groq-key/', store_groq_key, name='store_groq_key'),
    path('upload-pdf/<int:key_id>/', upload_pdf_and_create_bot, name='upload_pdf_and_create_bot'),
    path('get-bot-response/', get_bot_response, name='get_bot_response'),
]
