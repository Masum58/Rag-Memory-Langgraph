# =============================================================
# FILE: app/main.py
# =============================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers import chat, documents
from app.database import setup_tables

@asynccontextmanager
async def lifespan(app: FastAPI):
    # app চালু হলে tables বানাও
    await setup_tables()
    yield

app = FastAPI(lifespan=lifespan)

app.include_router(chat.router)
app.include_router(documents.router)