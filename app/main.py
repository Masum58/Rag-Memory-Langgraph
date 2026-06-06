# =============================================================
# FILE: app/main.py
# PURPOSE: FastAPI app setup
# =============================================================

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.routers import chat
from app.database import setup_memory_table

# =============================================================
# LIFESPAN
# PURPOSE: app চালু হলে table বানাও
# কেন lifespan: on_event deprecated, এটাই নতুন way
# =============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup — app চালু হলে চলবে
    await setup_memory_table()
    yield
    # shutdown — app বন্ধ হলে চলবে (এখন কিছু নেই)

app = FastAPI(lifespan=lifespan)

app.include_router(chat.router)