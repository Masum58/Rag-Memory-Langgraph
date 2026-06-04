from fastapi import APIRouter
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.graph import graph, builder
import os

router = APIRouter(prefix="/chat", tags=["chat"])

class ChatRequest(BaseModel):
    thread_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    DB_URI = os.getenv("DATABASE_URL")
    async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
        await checkpointer.setup()
        compiled_graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": request.thread_id}}
        result = await compiled_graph.ainvoke(
            {'messages': [HumanMessage(content=request.message)]},
            config=config
        )
    return ChatResponse(reply=result['messages'][-1].content)