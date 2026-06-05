# =============================================================
# FILE: app/graph.py
# PURPOSE: LangGraph graph বানানো এবং compile করা
# CALLED BY: app/routers/chat.py → from app.graph import graph, builder
# =============================================================

# trim_messages → conversation history trim করে token limit রাখে
# count_tokens_approximately → কতটুকু token আছে সেটা count করে
from langchain_core.messages import trim_messages
from langchain_core.messages.utils import count_tokens_approximately

# ChatOpenAI → OpenAI GPT model use করার জন্য
from langchain_openai import ChatOpenAI

# StateGraph → graph বানানোর main class
# START → graph কোথা থেকে শুরু হবে
# MessagesState → graph-এর shared notebook (messages list রাখে)
from langgraph.graph import StateGraph, START, MessagesState

from dotenv import load_dotenv
import os

# .env file থেকে OPENAI_API_KEY, DATABASE_URL ইত্যাদি load করো
load_dotenv()

# LLM object — graph-এর ভেতরে এটাই কাজ করবে
# DEFAULT: gpt-3.5-turbo, বদলাতে চাইলে → ChatOpenAI(model="gpt-4o-mini")
model = ChatOpenAI()

# =============================================================
# MAX_TOKENS: conversation history-র maximum token limit
# কেন: LLM-এ পুরো history পাঠালে cost বাড়ে, context limit ছাড়ায়
# বাড়াতে চাইলে: 1000, 2000 — কমালে কম history মনে থাকবে
# =============================================================
MAX_TOKENS = 500

# =============================================================
# FUNCTION: call_model
# PURPOSE: state থেকে messages নেয়, trim করে, LLM-কে দেয়
# PARAMETER: state → MessagesState (graph-এর shared notebook)
#   state["messages"] এ থাকে: [HumanMessage, AIMessage, ...]
# RETURNS: {"messages": [AIMessage]} → state-এ যোগ হয়
# CALLED BY: graph-এর ভেতরে automatically, invoke() করলে
# =============================================================
def call_model(state: MessagesState):

    # trim_messages → পুরো history না পাঠিয়ে শেষের 500 token রাখে
    # strategy="last" → শেষের messages রাখো, পুরনোগুলো বাদ দাও
    # token_counter → কতটুকু token সেটা count করার function
    # max_tokens → এর বেশি token পাঠাবে না
    # OUTPUT: trimmed messages list → [HumanMessage, AIMessage, ...]
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=MAX_TOKENS
    )

    # DEBUG: কতটুকু token আছে দেখতে চাইলে এই line uncomment করো
    # print('Current Token Count ->', count_tokens_approximately(messages=messages))

    # LLM-কে trimmed messages দাও, জবাব নাও
    # OUTPUT: AIMessage object → .content এ text থাকে
    response = model.invoke(messages)

    # state-এ নতুন AIMessage যোগ করো
    return {"messages": [response]}

# =============================================================
# GRAPH SETUP
# PURPOSE: node আর edge দিয়ে graph বানানো
# CALLED BY: এই file import হলে automatically চলে
# =============================================================

# MessagesState → graph-এর notebook type বলছো
builder = StateGraph(MessagesState)

# 'call_model' নামে একটা node যোগ করো → call_model function চালাবে
builder.add_node('call_model', call_model)

# START → call_model → END (সবসময় এই path)
builder.add_edge(START, 'call_model')

# graph compile করো — checkpointer chat.py তে দেওয়া হবে
# কেন এখানে checkpointer নেই: AsyncPostgresSaver async,
# তাই chat.py-তে request আসলে সেখানে initialize করা হয়
graph = builder.compile()

# DATABASE_URL → docker-compose.yml এর db service এর address
# FORMAT: postgresql://username:password@service_name:port/db_name
DB_URI = os.getenv("DATABASE_URL")