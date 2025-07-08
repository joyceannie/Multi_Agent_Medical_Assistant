"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

If you want to change the LLM and Embedding model:

you can do it by changing all 'llm' and 'embedding_model' variables present in multiple classes below.
"""

import os
from dotenv import load_dotenv
from langchain_openai.chat_models.base import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

class AgentDecisoinConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model = os.getenv("model_name"),  # Replace with your model name
            api_key = os.getenv("openai_api_key"),  # Replace with your  OpenAI API key
        )

class ConversationConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name = os.getenv("model_name"),  # Replace with your model name
            api_key = os.getenv("openai_api_key"),  # Replace with your OpenAI API key
            temperature = 0.7  # Creative but factual
        )

class WebSearchConfig:
    def __init__(self):
        self.llm = ChatOpenAI(
            model = os.getenv("model_name"),  # Replace with your model name
            api_key = os.getenv("openai_api_key"),  # Replace with your Azure OpenAI API key
            temperature = 0.3  # Slightly creative but factual
        )
        self.context_limit = 20        

