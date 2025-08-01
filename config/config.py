"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.
"""

import os
from dotenv import load_dotenv
from langchain_openai.chat_models.base import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


# Load environment variables from .env file
load_dotenv()

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model = os.getenv("MODEL_NAME"),  # Replace with your model name
            api_key = os.getenv("OPENAI_API_KEY"),  # Replace with your  OpenAI API key
        )

class ConversationConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name = os.getenv("MODEL_NAME"),  # Replace with your model name
            api_key = os.getenv("OPENAI_API_KEY"),  # Replace with your OpenAI API key
            temperature = 0.7  # Creative but factual
        )

class WebSearchConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model = os.getenv("MODEL_NAME"),  # Replace with your model name
            api_key = os.getenv("OPENAI_API_KEY"),  # Replace with your Azure OpenAI API key
            temperature = 0.3  # Slightly creative but factual
        )
        self.context_limit = 20     
        self.context_limit = 20    

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 1536  # Add the embedding dimension here
        self.distance_metric = "Cosine"  # Add this with a default value
        self.use_local = True  # Add this with a default value
        self.vector_local_path = "./data/qdrant_db"  # Add this with a default value
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag"  # Ensure a valid name
        self.chunk_size = 512  # Modify based on documents and performance
        self.chunk_overlap = 50  # Modify based on documents and performance
        # self.embedding_model = "text-embedding-3-large"
        # Initialize Azure OpenAI Embeddings
        self.embedding_model = OpenAIEmbeddings(
            model = os.getenv("EMBEDDING_MODEL_NAME"),  # Replace with your  model name
            api_key = os.getenv("OPENAI_API_KEY"),  # Replace with your  OpenAI API key
        )

        self.llm = ChatOpenAI(
            model = os.getenv("MODEL_NAME"),  # Replace with your model name
            api_key = os.getenv("OPENAI_API_KEY"),  # Replace with your  OpenAI API key
            temperature = 0.3  # Slightly creative but factual
        )
        self.summarizer_model = ChatOpenAI(
            model_name = os.getenv("MODEL_NAME"),  # Replace with your  model name
            api_key = os.getenv("OPENAI_API_KEY"),  # Replace with your OpenAI API key
            temperature = 0.5  # Slightly creative but factual
        )
        self.chunker_model = ChatOpenAI(
            model_name = os.getenv("MODEL_NAME"),  # Replace with your model name
            api_key = os.getenv("OPENAI_API_KEY"),  # Replace with your Azure OpenAI API key
            temperature = 0.0  # factual
        )
        self.response_generator_model = ChatOpenAI(
            model_name = os.getenv("MODEL_NAME"),  # Replace with your Azure model name
            api_key = os.getenv("OPENAI_API_KEY"),  # Replace with your Azure OpenAI API key
            temperature = 0.3  # Slightly creative but factual
        )
        self.top_k = 5
        self.vector_search_type = 'similarity'  # or 'mmr'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3

        self.max_context_length = 8192  # (Change based on your need) # 1024 proved to be too low (retrieved content length > context length = no context added) in formatting context in response_generator code

        self.include_sources = True  # Show links to reference documents and images along with corresponding query response

        # ADJUST ACCORDING TO ASSISTANT'S BEHAVIOUR BASED ON THE DATA INGESTED:
        self.min_retrieval_confidence = 0.40  # The auto routing from RAG agent to WEB_SEARCH agent is dependent on this value

        self.context_limit = 20     # include last 20 messsages (10 Q&A pairs) in history 

class InputGuardrailsConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model = os.getenv("MODEL_NAME"), 
            api_key = os.getenv("OPENAI_API_KEY"), 
            temperature = 0.0  # Factual and strict
        )

class OutputGuardrailsConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model = os.getenv("MODEL_NAME"),  
            api_key = os.getenv("OPENAI_API_KEY"),  
            temperature = 0.0  # Factual and strict
        )       

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.web_search = WebSearchConfig()
        self.rag = RAGConfig()
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20  # Include last 20 messsages (10 Q&A pairs) in history   

