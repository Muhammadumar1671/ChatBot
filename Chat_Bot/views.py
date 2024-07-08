from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .models import GroqKey
from .serializers import GroqKeySerializer
import hashlib
import os
from langchain_groq import ChatGroq
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

__retriver__ = None

class RetrieverSingleton:
    _instance = None
    retriever = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def initialize_retriever(self, docs):
        if not self.retriever:
            self.retriever = split_and_index_documents(docs)
        return self.retriever


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def store_groq_key(request):
    groq_key = request.data.get('groq_key')
    if not groq_key:
        return Response({'error': 'GROQ key is required'}, status=status.HTTP_400_BAD_REQUEST)
    
    groq_key_instance = GroqKey(user=request.user, hashed_key=groq_key)
    groq_key_instance.save()
    
    return Response({
        'message': 'GROQ key has been stored successfully',
        'id': groq_key_instance.id
    }, status=status.HTTP_201_CREATED)


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def upload_pdf_and_create_bot(request, key_id):
    try:
        groq_key_instance = GroqKey.objects.get(id=key_id, user=request.user)
    except GroqKey.DoesNotExist:
        return Response({'error': 'GROQ key not found'}, status=status.HTTP_404_NOT_FOUND)
    
    pdf_file = request.data.get('pdf_document')
    
    if not pdf_file:
        return Response({'error': 'PDF document is required'}, status=status.HTTP_400_BAD_REQUEST)
    
    groq_key_instance.pdf_document = pdf_file
    groq_key_instance.save()
    
    # Retrieve the hashed key from the database
    groq_key_instance = GroqKey.objects.get(id=key_id, user=request.user)

    answerFromBot = create_bot(groq_key_instance.hashed_key, groq_key_instance.pdf_document.path)
    
    return Response({'message': 'Bot initialized using PDF. Response from Bot:',
                     'response': answerFromBot},
                     status=status.HTTP_200_OK)


@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def get_bot_response(request):
    question = request.data.get('question')
    if not question:
        return Response({'error': 'Question is required'}, status=status.HTTP_400_BAD_REQUEST)

    retriever_singleton = RetrieverSingleton()
    if retriever_singleton.retriever is None:
        return Response({'error': 'Retriever not initialized'}, status=status.HTTP_400_BAD_REQUEST)

    prompt = define_rag_prompt()
    llm = initialize_chat_model("llama3-8b-8192")
    rag_chain = construct_rag_chain(retriever_singleton.retriever, llm, prompt)

    response = chatbot_interaction(rag_chain, question)

    return Response({'response': response}, status=status.HTTP_200_OK)


# Function to initialize environment with API key
def initialize_environment(api_key):
    os.environ["GROQ_API_KEY"] = api_key

# Function to load and split documents
def load_and_split_documents(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

# Initialize the ChatGroq model
def initialize_chat_model(model_name):
    return ChatGroq(model=model_name)

# Function to split and index contents
def split_and_index_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings())
    __retriver__ = vectorstore.as_retriever()
    return __retriver__

# Define a function to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG prompt
def define_rag_prompt():
    return hub.pull("rlm/rag-prompt")

# Construct the RAG chain for the ChatApp
def construct_rag_chain(retriever, llm, prompt):
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Function to interact with the chatbot
def chatbot_interaction(rag_chain, question):
    response = rag_chain.invoke(question)
    return response

def create_bot(hashed_key, pdf_path):
    # Take API key input
    api_key = hashed_key

    # Initialize environment with API key
    initialize_environment(api_key)

    # Take file path input
    file_path = pdf_path

    # Load and split documents
    docs = load_and_split_documents(file_path)

    # Initialize models
    llm = initialize_chat_model("llama3-8b-8192")

    # Use singleton to get or create the retriever
    retriever_singleton = RetrieverSingleton()
    retriever = retriever_singleton.initialize_retriever(docs)

    # Define RAG prompt
    prompt = define_rag_prompt()

    # Construct RAG chain
    rag_chain = construct_rag_chain(retriever, llm, prompt)

    user_input = "What is written in this document?"
    response = chatbot_interaction(rag_chain, user_input)

    return response
