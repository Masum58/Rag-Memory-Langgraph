# =============================================================
# FILE: app/routers/chat.py
# PURPOSE: /chat endpoint handle করা
# CALLED BY: app/main.py → app.include_router(chat.router)
# DEPENDS ON: app/graph.py → graph, builder
# =============================================================

# APIRouter → FastAPI-তে endpoint group করার জন্য
# prefix="/chat" মানে সব endpoint এ আগে /chat যোগ হবে
from fastapi import APIRouter

# BaseModel → request আর response এর shape define করার জন্য
from pydantic import BaseModel

# HumanMessage → user এর message কে LangChain format এ convert করে
from langchain_core.messages import HumanMessage

# AsyncPostgresSaver → PostgreSQL এ conversation save করার জন্য
# async কেন: FastAPI async, তাই database call ও async হওয়া দরকার
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# graph → compiled LangGraph graph
# builder → checkpointer দিয়ে আবার compile করার জন্য
from app.graph import graph, builder

import os

# =============================================================
# ROUTER SETUP
# prefix="/chat" → সব endpoint এ /chat আগে থাকবে
# tags=["chat"] → FastAPI docs এ "chat" গ্রুপে দেখাবে
# =============================================================
router = APIRouter(prefix="/chat", tags=["chat"])

# =============================================================
# REQUEST MODEL
# PURPOSE: user যা পাঠাবে তার shape
# FIELDS:
#   thread_id → কোন conversation এর memory তুলে আনবে
#   message → user এর কথা
# EXAMPLE: {"thread_id": "user-1", "message": "আমার নাম Masum"}
# =============================================================
class ChatRequest(BaseModel):
    thread_id: str
    message: str

# =============================================================
# RESPONSE MODEL
# PURPOSE: user কে যা পাঠাবে তার shape
# FIELDS:
#   reply → LLM এর জবাব
# EXAMPLE: {"reply": "Hello Masum!"}
# =============================================================
class ChatResponse(BaseModel):
    reply: str

# =============================================================
# ENDPOINT: POST /chat/
# PURPOSE: user এর message নেয়, graph চালায়, জবাব দেয়
# PARAMETER: request → ChatRequest (thread_id + message)
# RETURNS: ChatResponse (reply)
# CALLED BY: যেকোনো HTTP client → POST http://localhost:8000/chat/
# =============================================================
@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):

    # DATABASE_URL → .env থেকে আসে
    # FORMAT: postgresql://postgres:postgres@db:5432/ragmemory
    DB_URI = os.getenv("DATABASE_URL")

    # AsyncPostgresSaver → প্রতিটা request এ PostgreSQL connection খোলে
    # async with → request শেষে automatically connection বন্ধ করে
    # setup() → প্রথমবার চললে database এ table বানায়
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

        # builder → graph.py থেকে আনা, এখানে checkpointer দিয়ে compile
        # কেন এখানে compile: AsyncPostgresSaver async, graph.py তে করা যায় না
        compiled_graph = builder.compile(checkpointer=checkpointer)

        # thread_id → কোন conversation এর memory তুলে আনবে
        # একই thread_id দিলে আগের কথা মনে থাকবে
        config = {"configurable": {"thread_id": request.thread_id}}

        # graph চালাও
        # input: user এর message → HumanMessage format এ
        # config: thread_id দিয়ে memory খুঁজবে
        # OUTPUT: {"messages": [HumanMessage, AIMessage, ...]}
        result = await compiled_graph.ainvoke(
            {'messages': [HumanMessage(content=request.message)]},
            config=config
        )

    # result["messages"] → পুরো conversation history
    # [-1] → শেষেরটা নাও, এটাই LLM এর জবাব
    # .content → AIMessage এর text বের করো
    return ChatResponse(reply=result['messages'][-1].content)