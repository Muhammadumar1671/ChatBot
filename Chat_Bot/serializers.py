
from rest_framework import serializers
from .models import GroqKey

class GroqKeySerializer(serializers.ModelSerializer):
    class Meta:
        model = GroqKey
        fields = ['user', 'hashed_key', 'pdf_document']
        extra_kwargs = {'user': {'read_only': True}, 'hashed_key': {'write_only': True}}
