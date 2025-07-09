"""
FastAPI Server for Multi-Agent Medical Assistant

This module provides a RESTful API endpoint to interact with the LangGraph-based
medical assistant workflow, and now also serves the chat UI using Jinja2 templates.
"""

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import uvicorn
import os
from dotenv import load_dotenv

# Assuming your Langgraph workflow file is named 'agent_workflow.py'
# You would need to ensure this file is in the same directory or accessible via PYTHONPATH
from agent_workflow import create_agent_graph, AgentState # Import necessary components

load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Medical Assistant API",
    description="API for routing medical queries to specialized AI agents.",
    version="1.0.0",
)

# Configure Jinja2Templates
# Assuming your templates are in a directory named 'templates' in the same folder as main.py
templates = Jinja2Templates(directory="templates")

# In-memory storage for graph instance (for simplicity, in a real app, manage state carefully)
_langgraph_app = None

def get_langgraph_app():
    """Initializes and returns the LangGraph app instance."""
    global _langgraph_app
    if _langgraph_app is None:
        print("Initializing LangGraph application...")
        _langgraph_app = create_agent_graph()
        print("LangGraph application initialized.")
    return _langgraph_app

# Pydantic model for request body
class ChatRequest(BaseModel):
    """
    Represents the request body for a chat interaction.
    """
    query: Union[str, Dict[str, str]] = Field(
        ..., description="The user's input query. Can be a string or a dictionary for image input."
    )
    thread_id: Optional[str] = Field(
        "1", description="Unique identifier for the conversation thread. Defaults to '1'."
    )

class ChatResponse(BaseModel):
    """
    Represents the response body from a chat interaction.
    """
    response: str = Field(..., description="The AI's response to the query.")
    agent_used: Optional[str] = Field(
        None, description="The name of the agent that handled the query."
    )
    thread_id: str = Field(..., description="The thread ID of the conversation.")

@app.get("/", response_class=HTMLResponse, summary="Serve the chat UI")
async def read_root(request: Request):
    """
    Serves the main chat user interface.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat", response_model=ChatResponse, summary="Process a user query through the medical assistant")
async def chat_with_assistant(request: ChatRequest):
    """
    Receives a user query and routes it through the LangGraph-based medical assistant.
    The assistant will determine the appropriate agent (e.g., conversation, web search)
    to handle the request and return a response.
    """
    try:
        langgraph_app = get_langgraph_app()

        # Call the processing function from your Langgraph workflow
        # The `process_query` function in your `agent_workflow.py` returns the final `result` dictionary.
        # We need to extract the `output` and `agent_name` from this `result`.
        
        # NOTE: The `process_query` in your provided `agent_workflow.py` uses a fixed `thread_config`.
        # To make the FastAPI server support multiple concurrent conversations (threads),
        # you *must* modify `process_query` in `agent_workflow.py` to accept `thread_id`
        # and pass it to `graph.invoke(state, {"configurable": {"thread_id": thread_id}})`.
        # For this example, it will always use thread "1" for persistence.
        
        final_state = langgraph_app.invoke(
            input={
                "messages": [request.query if isinstance(request.query, str) else request.query.get("text", "")],
                "current_input": request.query,
                # "has_image": isinstance(request.query, dict) and "image" in request.query,
                # "image_type": request.query.get("image_type") if isinstance(request.query, dict) else None,
            },
            config={"configurable": {"thread_id": request.thread_id}}
        )

        response_content = final_state.get("output", "No response generated.")
        agent_name = final_state.get("agent_name", "Unknown")

        return ChatResponse(
            response=response_content,
            agent_used=agent_name,
            thread_id=request.thread_id
        )

    except Exception as e:
        print(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# Example of how to run the server (for development)
if __name__ == "__main__":
    # Ensure the LangGraph app is initialized when the server starts
    get_langgraph_app()
    # You can change the host and port as needed
    uvicorn.run(app, host="0.0.0.0", port=8000)
