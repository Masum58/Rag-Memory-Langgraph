# =============================================================
# FILE: app/routers/chat.py
# PURPOSE: /chat endpoint handle করা
# CALLED BY: app/main.py
# DEPENDS ON: app/graph.py → builder, ChatState
# =============================================================

from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import PostgresStore
from app.graph import builder, ChatState  # ChatState import করো
from app.database import get_user_documents
import os
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    thread_id: str
    user_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):

    logger.info(f"CHAT → thread_id: {request.thread_id} | user_id: {request.user_id}")

    DB_URI = os.getenv("DATABASE_URL")

    # user এর uploaded files list আনো
    # কেন এখানে: database call async, graph এ করা যায় না
    docs = await get_user_documents(request.user_id)
    files_list = "\n".join([
        f"- {doc['filename']} ({doc['chunks']} chunks)"
        for doc in docs
    ]) if docs else "(কোনো file upload করা নেই)"

    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()

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
                {
                    'messages': [HumanMessage(content=request.message)],
                    'uploaded_files': files_list,
                },
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