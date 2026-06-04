from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI()

def call_model(state: MessagesState):
    response = model.invoke(state['messages'])
    return {'messages': [response]}

builder = StateGraph(MessagesState)
builder.add_node('call_model', call_model)
builder.add_edge(START, 'call_model')

DB_URI = os.getenv("DATABASE_URL")

# graph আলাদা রাখো, checkpointer পরে দেবো
graph = builder.compile()
DB_URI = os.getenv("DATABASE_URL")