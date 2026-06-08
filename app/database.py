# =============================================================
# FILE: app/database.py
# PURPOSE: database table setup
# CALLED BY: app/main.py → lifespan
# =============================================================

import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

async def get_connection():
    return await asyncpg.connect(os.getenv("DATABASE_URL"))

# =============================================================
# FUNCTION: setup_tables
# PURPOSE: দুটো table বানানো — না থাকলে
# TABLES:
#   uploaded_documents → কোন user কোন file upload করেছে
# CALLED BY: app/main.py → lifespan startup
# =============================================================
async def setup_tables():
    conn = await get_connection()
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_documents (
            id          SERIAL PRIMARY KEY,
            user_id     TEXT NOT NULL,
            filename    TEXT NOT NULL,
            chunks      INTEGER NOT NULL,
            uploaded_at TIMESTAMP DEFAULT NOW()
        )
    """)
    await conn.close()

# =============================================================
# FUNCTION: save_document_record
# PURPOSE: upload হলে PostgreSQL এ record রাখো
# PARAMETER:
#   user_id  → কোন user এর
#   filename → কোন file
#   chunks   → কতটা chunk index হয়েছে
# CALLED BY: documents.py → upload_document
# =============================================================
async def save_document_record(user_id: str, filename: str, chunks: int):
    conn = await get_connection()
    await conn.execute("""
        INSERT INTO uploaded_documents (user_id, filename, chunks)
        VALUES ($1, $2, $3)
    """, user_id, filename, chunks)
    await conn.close()

# =============================================================
# FUNCTION: get_user_documents
# PURPOSE: একজন user এর সব uploaded documents আনা
# PARAMETER: user_id → যার documents চাই
# RETURNS: [{"filename": "...", "chunks": 10, "uploaded_at": "..."}]
# CALLED BY: documents.py → list_documents
# =============================================================
async def get_user_documents(user_id: str) -> list:
    conn = await get_connection()
    rows = await conn.fetch("""
        SELECT filename, chunks, uploaded_at
        FROM uploaded_documents
        WHERE user_id = $1
        ORDER BY uploaded_at DESC
    """, user_id)
    await conn.close()
    return [dict(row) for row in rows]