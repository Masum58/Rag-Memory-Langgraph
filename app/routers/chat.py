# =============================================================
# FILE: app/routers/chat.py
# PURPOSE: /chat endpoint handle করা
# CALLED BY: app/main.py
# DEPENDS ON: app/graph.py → builder
# =============================================================

from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import PostgresStore
from app.graph import builder
import os

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    thread_id: str
    user_id: str      # long-term memory এর জন্য
    message: str

class ChatResponse(BaseModel):
    reply: str

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):

    DB_URI = os.getenv("DATABASE_URL")

    # short-term memory → AsyncPostgresSaver
    # long-term memory → PostgresStore
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

        # PostgresStore → sync, context manager দিয়ে
        with PostgresStore.from_conn_string(DB_URI) as store:
            store.setup()

            compiled_graph = builder.compile(
                checkpointer=checkpointer,
                store=store
            )

            config = {
                "configurable": {
                    "thread_id": request.thread_id,
                    "user_id": request.user_id,
                }
            }

            result = await compiled_graph.ainvoke(
                {'messages': [HumanMessage(content=request.message)]},
                config=config
            )

    return ChatResponse(reply=result['messages'][-1].content)

# =============================================================
# ENDPOINT: GET /chat/conversations
# PURPOSE: সব saved conversation list দেওয়া
# =============================================================
@router.get("/conversations")
async def get_conversations():

    DB_URI = os.getenv("DATABASE_URL")

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

        async with checkpointer.conn.cursor() as cur:
            await cur.execute(
                "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
            )
            rows = await cur.fetchall()

    conversations = [row["thread_id"] for row in rows]
    return {"conversations": conversations}