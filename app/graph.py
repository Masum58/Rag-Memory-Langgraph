# =============================================================
# FILE: app/graph.py
# PURPOSE: LangGraph graph — short-term + long-term memory সহ
# CALLED BY: app/routers/chat.py
# =============================================================

import uuid
from typing import List, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.base import BaseStore
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI()
memory_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# =============================================================
# STATE: ChatState
# PURPOSE: graph এর shared notebook
# FIELDS:
#   messages → conversation history
#   summary  → পুরনো conversation এর compressed summary
# NOTE: user_id এখন config থেকে আসে, state এ নেই
# =============================================================
class ChatState(MessagesState):
    summary: str = ""

# =============================================================
# MEMORY MODELS
# PURPOSE: structured output দিয়ে de-duplication করা
# =============================================================
class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if NEW info, False if duplicate")

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)

memory_extractor = memory_llm.with_structured_output(MemoryDecision)

# =============================================================
# PROMPTS
# =============================================================
MEMORY_PROMPT = """You are responsible for updating and maintaining accurate user memory.

CURRENT USER DETAILS (existing memories):
{user_details_content}

TASK:
- Review the user's latest message.
- Extract user-specific info worth storing long-term (identity, stable preferences, ongoing projects/goals).
- For each extracted item, set is_new=true ONLY if it adds NEW information.
- If same meaning already present, set is_new=false.
- Keep each memory as a short atomic sentence.
- No speculation; only facts stated by the user.
- If nothing memory-worthy, return should_write=false and empty list.
"""

SYSTEM_PROMPT = """তুমি একজন helpful assistant যার memory আছে।
user যে ভাষায় কথা বলবে, তুমি সেই ভাষায় reply করবে।

যদি user সম্পর্কে তথ্য জানা থাকে, সেটা দিয়ে personalize করো:
- নাম জানলে নাম ধরে ডাকো
- পেশা বা project জানলে সেটা reference করো

User সম্পর্কে যা জানা আছে:
{user_details_content}
"""

# =============================================================
# FUNCTION: remember_node
# PURPOSE: conversation থেকে নতুন তথ্য extract করে store এ save করা
# PARAMETER:
#   state → ChatState
#   config → user_id এখানে থাকে
#   store → PostgresStore — LangGraph automatically inject করে
# RETURNS: {} — state বদলায় না
# =============================================================
def remember_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):

    # user_id → config থেকে আসে
    # কেন config: একই graph বিভিন্ন user এর জন্য চলে
    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    # PostgresStore থেকে existing memories আনো
    items = store.search(ns)
    existing = "\n".join(
        it.value.get("data", "") for it in items
    ) if items else "(empty)"

    # latest user message
    last_text = state["messages"][-1].content

    # LLM দিয়ে নতুন তথ্য extract করো
    decision: MemoryDecision = memory_extractor.invoke([
        SystemMessage(content=MEMORY_PROMPT.format(
            user_details_content=existing
        )),
        {"role": "user", "content": last_text},
    ])

    # শুধু নতুন তথ্য save করো — duplicate হবে না
    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                # uuid4() → unique key, duplicate key হবে না
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})

    return {}

# =============================================================
# FUNCTION: chat_node
# PURPOSE: memory + summary দিয়ে LLM call করা
# PARAMETER:
#   state → ChatState
#   config → user_id এখানে থাকে
#   store → PostgresStore — LangGraph automatically inject করে
# RETURNS: {"messages": [AIMessage]}
# =============================================================
def chat_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):

    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    # Long-term memory load করো
    items = store.search(ns)
    user_details = "\n".join(
        it.value.get("data", "") for it in items
    ) if items else "(empty)"

    summary = state.get("summary", "")

    messages = []

    # System prompt — personalization + language
    messages.append(SystemMessage(
        content=SYSTEM_PROMPT.format(user_details_content=user_details)
    ))

    # Short-term summary থাকলে যোগ করো
    if summary:
        messages.append(SystemMessage(
            content=f"Conversation summary:\n{summary}"
        ))

    messages.extend(state["messages"])

    response = model.invoke(messages)
    return {"messages": [response]}

# =============================================================
# FUNCTION: summarize_node
# PURPOSE: পুরনো messages summary করে delete করা
# =============================================================
def summarize_node(state: ChatState):

    existing_summary = state.get("summary", "")

    if existing_summary:
        prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            "Extend the summary using the new conversation above."
        )
    else:
        prompt = "Summarize the conversation above."

    messages_for_summary = state["messages"] + [
        HumanMessage(content=prompt)
    ]
    response = model.invoke(messages_for_summary)

    messages_to_delete = state["messages"][:-2]

    return {
        "summary": response.content,
        "messages": [RemoveMessage(id=m.id) for m in messages_to_delete]
    }

# =============================================================
# FUNCTION: should_summarize
# PURPOSE: 10 এর বেশি messages হলে summarize করো
# =============================================================
def should_summarize(state: ChatState) -> Literal["summarize", END]:
    if len(state["messages"]) > 10:
        return "summarize"
    return END

# =============================================================
# GRAPH SETUP
# FLOW:
#   START → remember → chat → should_summarize?
#                               → True → summarize → END
#                               → False → END
# =============================================================
builder = StateGraph(ChatState)

builder.add_node("remember", remember_node)
builder.add_node("chat", chat_node)
builder.add_node("summarize", summarize_node)

builder.add_edge(START, "remember")
builder.add_edge("remember", "chat")
builder.add_conditional_edges("chat", should_summarize)
builder.add_edge("summarize", END)

graph = builder.compile()

DB_URI = os.getenv("DATABASE_URL")