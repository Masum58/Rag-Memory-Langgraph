# =============================================================
# FILE: app/graph.py
# PURPOSE: LangGraph graph বানানো — summarization + remove সহ
# CALLED BY: app/routers/chat.py → from app.graph import builder
# =============================================================

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from typing import Literal
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI()

# =============================================================
# STATE: ChatState
# PURPOSE: MessagesState extend করে summary field যোগ করা
# FIELDS:
#   messages → conversation history [HumanMessage, AIMessage, ...]
#   summary  → পুরনো conversation এর compressed summary
#              DEFAULT: "" — প্রথমবার কোনো summary নেই
# =============================================================
class ChatState(MessagesState):
    summary: str = ""

# =============================================================
# FUNCTION: chat_node
# PURPOSE: summary + নতুন messages মিলিয়ে LLM call করা
# PARAMETER: state → ChatState
#   state["messages"] → নতুন messages
#   state["summary"]  → আগের summary (থাকলে)
# RETURNS: {"messages": [AIMessage]}
# CALLED BY: graph automatically — invoke() করলে
# =============================================================
def chat_node(state: ChatState):

    # .get() দিয়ে check করো — summary না থাকলে "" return করবে
    summary = state.get("summary", "")

    # summary থাকলে SystemMessage হিসেবে আগে রাখো
    # কেন: LLM জানবে আগে কী হয়েছিলো
    messages = []
    if summary:
        messages.append(SystemMessage(
            content=f"Conversation summary:\n{summary}"
        ))

    # summary এর পরে নতুন messages যোগ করো
    messages.extend(state["messages"])

    # LLM call করো
    response = model.invoke(messages)

    return {"messages": [response]}

# =============================================================
# FUNCTION: summarize_node
# PURPOSE: পুরনো messages summary করে delete করা
# PARAMETER: state → ChatState
# RETURNS: {"summary": নতুন summary, "messages": [RemoveMessage...]}
# CALLED BY: graph automatically — should_summarize True হলে
# =============================================================
def summarize_node(state: ChatState):

    # .get() দিয়ে check করো — summary না থাকলে "" return করবে
    existing_summary = state.get("summary", "")

    # আগের summary থাকলে extend করো, না থাকলে নতুন বানাও
    # কেন: প্রতিবার পুরো summary নতুন করে বানানো costly
    if existing_summary:
        prompt = (
            f"Existing summary:\n{existing_summary}\n\n"
            "Extend the summary using the new conversation above."
        )
    else:
        prompt = "Summarize the conversation above."

    # সব messages + summary prompt LLM কে দাও
    messages_for_summary = state["messages"] + [
        HumanMessage(content=prompt)
    ]
    response = model.invoke(messages_for_summary)

    # শেষের 2টা message রাখো, বাকি সব delete করো
    # কেন 2টা রাখি: recent context থাকা দরকার
    messages_to_delete = state["messages"][:-2]

    return {
        "summary": response.content,
        # RemoveMessage → LangGraph কে বলছো এই id এর message delete করো
        "messages": [RemoveMessage(id=m.id) for m in messages_to_delete]
    }

# =============================================================
# FUNCTION: should_summarize
# PURPOSE: summarize_node চালাবো কিনা সেটা decide করা
# PARAMETER: state → ChatState
# RETURNS: "summarize" অথবা END
# CALLED BY: graph — chat_node এর পরে conditional edge হিসেবে
# LOGIC: 10 এর বেশি messages হলে summarize করো
# =============================================================
def should_summarize(state: ChatState) -> Literal["summarize", END]:
    if len(state["messages"]) > 10:
        return "summarize"
    return END

# =============================================================
# GRAPH SETUP
# FLOW: START → chat_node → should_summarize?
#                               → True  → summarize_node → END
#                               → False → END
# =============================================================
builder = StateGraph(ChatState)

builder.add_node("chat", chat_node)
builder.add_node("summarize", summarize_node)

builder.add_edge(START, "chat")

# chat_node এর পরে should_summarize check করো
builder.add_conditional_edges("chat", should_summarize)

# summarize_node শেষ হলে END
builder.add_edge("summarize", END)

# graph compile করো — checkpointer chat.py তে দেওয়া হবে
graph = builder.compile()

DB_URI = os.getenv("DATABASE_URL")