# =============================================================
# FILE: app/database.py
# PURPOSE: user_memories table বানানো এবং CRUD operations
# CALLED BY: app/graph.py → memory load/save করতে
# =============================================================

import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()

# =============================================================
# FUNCTION: get_connection
# PURPOSE: PostgreSQL connection নেওয়া
# RETURNS: asyncpg connection object
# =============================================================
async def get_connection():
    return await asyncpg.connect(os.getenv("DATABASE_URL"))

# =============================================================
# FUNCTION: setup_memory_table
# PURPOSE: user_memories table বানানো — না থাকলে
# TABLE STRUCTURE:
#   user_id   → কোন user এর memory (thread_id এর prefix)
#   key       → memory এর নাম, যেমন "name", "city"
#   value     → memory এর value, যেমন "Masum", "Dhaka"
#   updated_at → কখন update হয়েছে
# CALLED BY: chat.py → startup এ একবার
# =============================================================
async def setup_memory_table():
    conn = await get_connection()
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS user_memories (
            user_id    TEXT NOT NULL,
            key        TEXT NOT NULL,
            value      TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT NOW(),
            PRIMARY KEY (user_id, key)
        )
    """)
    await conn.close()

# =============================================================
# FUNCTION: get_memories
# PURPOSE: একজন user এর সব memory আনা
# PARAMETER: user_id → যার memory চাই
# RETURNS: {"name": "Masum", "city": "Dhaka", ...}
# CALLED BY: graph.py → memory_load_node
# =============================================================
async def get_memories(user_id: str) -> dict:
    conn = await get_connection()
    rows = await conn.fetch(
        "SELECT key, value FROM user_memories WHERE user_id = $1",
        user_id
    )
    await conn.close()
    # rows → [{"key": "name", "value": "Masum"}, ...]
    # dict comprehension দিয়ে → {"name": "Masum", ...}
    return {row["key"]: row["value"] for row in rows}

# =============================================================
# FUNCTION: save_memories
# PURPOSE: memory save বা update করা — duplicate হবে না
# PARAMETER:
#   user_id  → যার memory
#   memories → {"name": "Masum", "city": "Dhaka"}
# CALLED BY: graph.py → memory_extract_node
# ON CONFLICT → same key থাকলে update করো, duplicate না
# =============================================================
async def save_memories(user_id: str, memories: dict):
    conn = await get_connection()
    for key, value in memories.items():
        await conn.execute("""
            INSERT INTO user_memories (user_id, key, value, updated_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT (user_id, key)
            DO UPDATE SET value = $3, updated_at = NOW()
        """, user_id, key, value)
    await conn.close()