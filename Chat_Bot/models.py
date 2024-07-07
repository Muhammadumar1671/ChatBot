# Chat_Bot/models.py

from django.db import models
from django.contrib.auth.models import User

class GroqKey(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    hashed_key = models.CharField(max_length=255)
    pdf_document = models.FileField(upload_to='pdf_documents/', null=True, blank=True)

    def __str__(self):
        return f"{self.user.username}'s GROQ Key"
