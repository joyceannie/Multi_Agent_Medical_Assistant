# Multi Agent Medical Assistant

A modular, agentic medical assistant powered by FastAPI, LangGraph, and modern LLMs. This application intelligently routes user queries to specialized agents (such as conversational, web search, and RAG agents) and supports both text and image-based queries.

## Features

- **Agentic Routing:** Dynamically selects the best agent for each query (conversation, web search, or retrieval-augmented generation).
- **Web UI:** Clean, responsive chat interface built with Tailwind CSS and Jinja2 templates.
- **REST API:** FastAPI backend for easy integration and extension.
- **Guardrails:** Input and output guardrails for safety and compliance.
- **Threaded Conversations:** Supports multi-turn conversations with thread IDs.

## Project Structure

```
.
├── app.py                  # FastAPI server entry point
├── agents/                 # All agent logic and orchestration
│   ├── agent_workflow.py   # Main agent routing and workflow logic
│   ├── guardrails/         # Input/output guardrails
│   ├── rag_agent/          # Retrieval-augmented generation agent
│   └── web_search_agent/   # Web search agent
├── config/                 # Configuration files
├── templates/              # Jinja2 HTML templates (UI)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
└── README.md               # Project documentation
```

## Getting Started

### 1. Install Dependencies

```sh
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create a `.env` file in the project root. Add any required API keys or configuration variables (see `config/config.py` for details).

### 3. Run the Server

```sh
python app.py
```

The server will start at [http://localhost:8000](http://localhost:8000).

### 4. Access the Chat UI

Open your browser and go to [http://localhost:8000](http://localhost:8000) to use the medical assistant chat interface.

## API Usage

- **POST `/chat`**  
  Send a user query and receive a response from the assistant.

  **Request Body:**
  ```json
  {
    "query": "What are the symptoms of flu?",
    "thread_id": "1"
  }
  ```

  **Response:**
  ```json
  {
    "response": "The symptoms of flu include...",
    "agent_used": "CONVERSATION_AGENT",
    "thread_id": "1"
  }
  ```

  ## Agents Overview

- **Conversation Agent:** Handles general chat, greetings, and non-medical questions.
- **Web Search Agent:** Answers questions about recent medical developments or outbreaks.
- **RAG Agent:** (Retrieval-Augmented Generation) Handles knowledge-intensive queries using document retrieval.

See [agents/README.md](agents/README.md) for more details on each agent.

## Customization

- **Add new agents:** Implement your agent in the `agents/` directory and update the routing logic in [`agents/agent_workflow.py`](agents/agent_workflow.py).
- **Modify prompts or thresholds:** Edit the `AgentConfig` class in [`agents/agent_workflow.py`](agents/agent_workflow.py).
- **UI changes:** Edit `templates/index.html`.

## License

This project is for educational and research purposes only. Not for clinical use.
