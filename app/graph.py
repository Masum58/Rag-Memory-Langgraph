# =============================================================
# FILE: app/graph.py
# PURPOSE: LangGraph graph — short-term + long-term memory + RAG
# CALLED BY: app/routers/chat.py
# FLOW:
#   START → remember → retrieval → chat → should_summarize?
#                                           → True → summarize → END
#                                           → False → END
# =============================================================

import uuid
from typing import List, Literal
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, RemoveMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.store.base import BaseStore
from dotenv import load_dotenv
import os

load_dotenv()

# =============================================================
# MODULE LEVEL SETUP — একবারই চলে, প্রতিটা request এ না
# কেন: প্রতিবার নতুন connection বানানো costly
# =============================================================
model = ChatOpenAI()
memory_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Pinecone vector store — module load হলে একবার initialize
# কেন module level: chat_node এ প্রতিবার বানালে প্রতিটা
# request এ নতুন connection খোলে, এটা avoid করা হচ্ছে
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ragmemory")
vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# =============================================================
# STATE: ChatState
# PURPOSE: graph এর shared notebook
# FIELDS:
#   messages       → conversation history [HumanMessage, AIMessage]
#   summary        → পুরনো conversation এর compressed summary
#   retrieved_docs → Pinecone থেকে আসা relevant chunks
# NOTE: user_id config থেকে আসে, state এ নেই
# =============================================================
class ChatState(MessagesState):
    summary: str = ""
    retrieved_docs: str = ""  # retrieval_node এখানে লিখবে

# =============================================================
# MEMORY MODELS
# PURPOSE: structured output দিয়ে de-duplication করা
# FIELDS:
#   text   → atomic memory sentence
#   is_new → True হলে save করবো, False হলে duplicate, skip করবো
# =============================================================
class MemoryItem(BaseModel):
    text: str = Field(description="Atomic user memory as a short sentence")
    is_new: bool = Field(description="True if NEW info, False if duplicate")

class MemoryDecision(BaseModel):
    should_write: bool
    memories: List[MemoryItem] = Field(default_factory=list)

# with_structured_output → LLM এর output MemoryDecision format এ আসবে
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

SYSTEM_PROMPT = """তুমি একজন helpful AI assistant যার memory এবং document knowledge base আছে।
user যে ভাষায় কথা বলবে, তুমি সেই ভাষায় reply করবে।

Document knowledge base থেকে পাওয়া প্রাসঙ্গিক তথ্য (Context):
{document_context}

User সম্পর্কে personal তথ্য (User Memory):
{user_details_content}

গুরুত্বপূর্ণ নির্দেশনা:
1. Context এ যে তথ্য আছে তা user এর upload করা document থেকে আনা হয়েছে।
2. User যদি document এর বিষয়ে জিজ্ঞেস করে, Context থেকে সঠিক তথ্য দাও।
3. Context এ উত্তর না থাকলে নিজের জ্ঞান থেকে উত্তর দাও।
4. কখনো বলবে না "I don't have access to files"।
"""

# =============================================================
# FUNCTION: remember_node
# PURPOSE: conversation থেকে নতুন তথ্য extract করে PostgresStore এ save
# PARAMETER:
#   state  → ChatState
#   config → user_id এখানে থাকে
#   store  → PostgresStore — LangGraph automatically inject করে
# RETURNS: {} — state বদলায় না, শুধু database এ save করে
# CALLED BY: graph → START এর পরে
# =============================================================
def remember_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):

    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    # PostgresStore থেকে existing memories আনো
    items = store.search(ns)
    existing = "\n".join(
        it.value.get("data", "") for it in items
    ) if items else "(empty)"

    last_text = state["messages"][-1].content

    # LLM দিয়ে নতুন তথ্য extract করো
    # OUTPUT: MemoryDecision object
    decision: MemoryDecision = memory_extractor.invoke([
        SystemMessage(content=MEMORY_PROMPT.format(
            user_details_content=existing
        )),
        {"role": "user", "content": last_text},
    ])

    # শুধু নতুন তথ্য save করো
    # is_new=False → duplicate, skip করো
    if decision.should_write:
        for mem in decision.memories:
            if mem.is_new and mem.text.strip():
                store.put(ns, str(uuid.uuid4()), {"data": mem.text.strip()})

    return {}

# =============================================================
# FUNCTION: retrieval_node
# PURPOSE: user এর নিজের namespace থেকে relevant docs আনা
# PARAMETER: state → ChatState, config → user_id এখানে থাকে
# RETURNS: {"retrieved_docs": "chunk1\n---\nchunk2"}
# CALLED BY: graph → remember_node এর পরে
# =============================================================
def retrieval_node(state: ChatState, config: RunnableConfig):

    user_id = config["configurable"]["user_id"]
    last_user_message = state["messages"][-1].content

    try:
        # namespace=user_id → শুধু এই user এর documents থেকে search
        # কেন namespace: অন্য user এর document মিশবে না
        user_vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=user_id
        )
        docs = user_vector_store.similarity_search(last_user_message, k=3)

        if docs:
            retrieved = "\n---\n".join([d.page_content for d in docs])
        else:
            retrieved = "(কোনো প্রাসঙ্গিক document পাওয়া যায়নি)"

    except Exception as e:
        retrieved = f"(document search এ সাময়িক সমস্যা: {str(e)})"

    return {"retrieved_docs": retrieved}

# =============================================================
# FUNCTION: chat_node
# PURPOSE: memory + retrieved_docs + summary দিয়ে LLM call করা
# PARAMETER:
#   state  → ChatState (retrieved_docs এখানে আছে)
#   config → user_id এখানে থাকে
#   store  → PostgresStore — LangGraph automatically inject করে
# RETURNS: {"messages": [AIMessage]}
# CALLED BY: graph → retrieval_node এর পরে
# =============================================================
def chat_node(state: ChatState, config: RunnableConfig, *, store: BaseStore):

    user_id = config["configurable"]["user_id"]
    ns = ("user", user_id, "details")

    # Long-term memory load করো
    items = store.search(ns)
    user_details = "\n".join(
        it.value.get("data", "") for it in items
    ) if items else "(empty)"

    # retrieval_node এর output — state থেকে পড়ো
    document_context = state.get("retrieved_docs", "(কোনো document নেই)")

    summary = state.get("summary", "")

    messages = []

    # System prompt — memory + document context + language
    messages.append(SystemMessage(
        content=SYSTEM_PROMPT.format(
            user_details_content=user_details,
            document_context=document_context
        )
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
# PARAMETER: state → ChatState
# RETURNS: {"summary": নতুন summary, "messages": [RemoveMessage...]}
# CALLED BY: graph → should_summarize True হলে
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

    # শেষের 2টা message রাখো, বাকি delete করো
    messages_to_delete = state["messages"][:-2]

    return {
        "summary": response.content,
        "messages": [RemoveMessage(id=m.id) for m in messages_to_delete]
    }

# =============================================================
# FUNCTION: should_summarize
# PURPOSE: 10 এর বেশি messages হলে summarize করো
# RETURNS: "summarize" অথবা END
# CALLED BY: graph → chat_node এর পরে conditional edge হিসেবে
# =============================================================
def should_summarize(state: ChatState) -> Literal["summarize", END]:
    if len(state["messages"]) > 10:
        return "summarize"
    return END

# =============================================================
# GRAPH SETUP
# FLOW:
#   START
#     → remember      (long-term memory extract + save)
#     → retrieval     (Pinecone থেকে relevant docs আনো)
#     → chat          (memory + docs + summary → LLM)
#     → should_summarize?
#         → True  → summarize → END
#         → False → END
# =============================================================
builder = StateGraph(ChatState)

builder.add_node("remember", remember_node)
builder.add_node("retrieval", retrieval_node)
builder.add_node("chat", chat_node)
builder.add_node("summarize", summarize_node)

builder.add_edge(START, "remember")
builder.add_edge("remember", "retrieval")
builder.add_edge("retrieval", "chat")
builder.add_conditional_edges("chat", should_summarize)
builder.add_edge("summarize", END)

graph = builder.compile()

DB_URI = os.getenv("DATABASE_URL")