# Agents

Here is a brief overview of the multiple agents used in the medical assistant project.

1. Web Search Agent
The web search agent is used to answer queries using web searches. Tavily search is performed to get relevant results from the web. Thses results are then given to the LLM. The LLM creates a summary based on the results, and sends the answer back to the user.

2. Conversational Agent
The conversational agent uses the underlying LLM to generate responses for general queries like greeting that doesn't need any specialized knowledge. 

3. RAG Agent
