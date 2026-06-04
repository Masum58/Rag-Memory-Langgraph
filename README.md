# RAG Memory LangGraph

A conversational AI API with persistent memory, built with LangGraph, FastAPI, and PostgreSQL.

## Features

- 🧠 **Persistent Memory** — Conversations are saved in PostgreSQL, so memory survives container restarts
- 🔗 **LangGraph** — Stateful conversation graph
- ⚡ **FastAPI** — REST API with automatic docs
- 🐳 **Docker** — Fully containerized
- 📊 **LangSmith** — Conversation tracing and monitoring

## Project Structure

├── app/
│   ├── main.py          # FastAPI app
│   ├── graph.py         # LangGraph graph
│   └── routers/
│       └── chat.py      # Chat endpoint
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/Masum58/rag-memory-langgraph.git
cd rag-memory-langgraph
```

### 2. Environment setup

```bash
cp .env.example .env
```

`.env` file-এ তোমার API keys দাও:
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key

### 3. Docker দিয়ে চালাও

```bash
docker-compose up --build
```

### 4. API test করো

Browser-এ যাও:
http://localhost:8000/docs

## API Usage

**POST** `/chat/`

```json
{
  "thread_id": "user-1",
  "message": "আমার নাম Masum"
}
```

Response:

```json
{
  "reply": "Hello Masum! How can I help you?"
}
```

Same `thread_id` দিয়ে পরের request করলে আগের conversation মনে থাকবে।

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [FastAPI](https://fastapi.tiangolo.com)
- [PostgreSQL](https://www.postgresql.org)
- [Docker](https://www.docker.com)
- [LangSmith](https://smith.langchain.com)