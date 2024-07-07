
from django.urls import path
from .views import store_groq_key, upload_pdf_and_create_bot

urlpatterns = [
    path('store-groq-key/', store_groq_key, name='store_groq_key'),
    path('upload-pdf/<int:key_id>/', upload_pdf_and_create_bot, name='upload_pdf_and_create_bot'),
]
