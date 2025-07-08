import os
from .tavily_search import TavilySearchAgent
from typing import Dict, List, Optional
from dotenv import load_dotenv
from config.config import WebSearchConfig

load_dotenv()

class WebSearchProcessor:
    """
    Processes web search results and routes them to the appropriate LLM for response generation.
    """
    
    def __init__(self, config):
        self.web_search_agent = TavilySearchAgent()
        
        # Initialize LLM for processing web search results
        self.llm = config.llm
    
    def _build_prompt_for_web_search(self, query: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        Build the prompt for the web search.
        
        Args:
            query: User query
            chat_history: chat history
            
        Returns:
            Complete prompt string
        """
        # Add chat history if provided
        # print("Chat History:", chat_history)
            
        # Build the prompt
        prompt = f"""Here are the last few messages from our conversation:

        {chat_history}

        The user asked the following question:

        {query}

        Summarize them into a single, well-formed question only if the past conversation seems relevant to the current query so that it can be used for a web search.
        Keep it concise and ensure it captures the key intent behind the discussion.
        """

        return prompt
    
    def process_web_results(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Fetches web search results, processes them using LLM, and returns a user-friendly response.
        """
        web_search_query_prompt = self._build_prompt_for_web_search(query=query, chat_history=chat_history)
        web_search_query = self.llm.invoke(web_search_query_prompt)
        
        # Retrieve web search results
        web_results = self.web_search_agent.search_tavily(web_search_query.content)

        
        # Construct prompt to LLM for processing the results
        llm_prompt = (
            "You are an AI assistant specialized in medical information. Below are web search results "
            "retrieved for a user query. Summarize and generate a helpful, concise response. "
            "Use reliable sources only and ensure medical accuracy.\n\n"
            f"Query: {query}\n\nWeb Search Results:\n{web_results}\n\nResponse:"
        )
        
        # Invoke the LLM to process the results
        response = self.llm.invoke(llm_prompt)
        
        return response
    
if __name__ == "__main__":
    websearch_config = WebSearchConfig()
    websearch_agent = WebSearchProcessor(websearch_config)
    response = websearch_agent.process_web_results(query="Tell me about nipah virus")
    print(response)
