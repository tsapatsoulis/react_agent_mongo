# ReActAgent with MongoDB Persistence

This repository contains a production-grade Python ReAct agent with official MongoDB checkpointing and session management, suitable for advanced LLM workflows.

## Features
- ReAct (Reasoning and Acting) agent pattern
- LangGraph stateful workflow
- MongoDB-based checkpointing for persistence and recovery
- Example tools: knowledge base search, math evaluation, timestamp
- Robust session management and analytics

## Usage
1. Copy `react_agent.py` and `.env.agents` to your project directory.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   # or manually:
   pip install langchain langgraph langchain-openai python-dotenv pymongo
   ```
3. Set your OpenAI API key in `.env.agents`:
   ```
   OPENAI_KEY=sk-xxxxxxx
   ```
4. Start MongoDB (local or Docker):
   ```bash
   docker run -d --name mongodb -p 27017:27017 mongo:7
   ```
5. Run the agent:
   ```bash
   python react_agent.py
   ```

## Example
The agent demonstrates multi-turn reasoning, tool use, and session stats with MongoDB.

## License
MIT
