# =============================================================
# FILE: streamlit_app.py
# PURPOSE: Streamlit UI — FastAPI /chat endpoint call করে
# RUNS ON: http://localhost:8501
# DEPENDS ON: FastAPI api service — http://api:8000/chat/
# =============================================================

import streamlit as st
import requests
import uuid
from datetime import datetime

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="RAG Memory Chat",
    page_icon="🧠",
    layout="wide"
)

# =============================================================
# CUSTOM CSS
# =============================================================
st.markdown("""
    <style>
        .user-bubble {
            background-color: #DCF8C6;
            padding: 10px 15px;
            border-radius: 15px 15px 0px 15px;
            margin: 5px 0;
            max-width: 70%;
            margin-left: auto;
            color: black;
        }
        .bot-bubble {
            background-color: #F1F0F0;
            padding: 10px 15px;
            border-radius: 15px 15px 15px 0px;
            margin: 5px 0;
            max-width: 70%;
            color: black;
        }
        .timestamp {
            font-size: 10px;
            color: #999;
            margin-top: 2px;
        }
        .user-timestamp {
            text-align: right;
        }
        .thinking {
            background-color: #F1F0F0;
            padding: 10px 15px;
            border-radius: 15px 15px 15px 0px;
            max-width: 70%;
            color: #999;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# =============================================================
# SESSION STATE
# PURPOSE: browser refresh না হওয়া পর্যন্ত data রাখা
# FIELDS:
#   messages    → UI তে দেখানোর জন্য conversation history
#   thread_id   → এই session এর unique id — memory track করতে
#   is_thinking → LLM call চলছে কিনা
#   user_id     → long-term memory এর জন্য user identifier
# =============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False

if "user_id" not in st.session_state:
    # প্রতিটা browser session এ একটাই user_id
    # thread_id বদলালেও user_id same থাকে
    # কেন: একই user অনেক conversation করতে পারে
    st.session_state.user_id = str(uuid.uuid4())

# =============================================================
# SIDEBAR
# PURPOSE: conversation management
# =============================================================
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    st.subheader("💬 Conversations")

    # PostgreSQL থেকে conversation list আনো
    try:
        res = requests.get("http://api:8000/chat/conversations")
        conversations = res.json()["conversations"]
    except:
        conversations = []

    # current thread_id list এ না থাকলে যোগ করো
    if st.session_state.thread_id not in conversations:
        conversations = [st.session_state.thread_id] + conversations

    # New Conversation button
    if st.button("🆕 New Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    # নিজে ID লিখে দেওয়ার option
    custom_id = st.text_input(
        "অথবা নিজে ID লেখো",
        placeholder="thread-id লেখো...",
    )
    if st.button("Load", use_container_width=True):
        if custom_id:
            st.session_state.thread_id = custom_id
            st.session_state.messages = []
            st.rerun()

    st.divider()

    # Conversation dropdown
    if conversations:
        current_index = (
            conversations.index(st.session_state.thread_id)
            if st.session_state.thread_id in conversations
            else 0
        )

        selected = st.selectbox(
            "Saved Conversations",
            options=conversations,
            index=current_index,
            format_func=lambda x: f"💬 {x[:8]}..."
        )

        if selected != st.session_state.thread_id:
            st.session_state.thread_id = selected
            st.session_state.messages = []
            st.rerun()

    else:
        st.caption("কোনো saved conversation নেই।")

    if st.button("🔄 Refresh List", use_container_width=True):
        st.rerun()

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("📊 Stats")
    st.metric("Messages", len(st.session_state.messages))
    st.caption(f"Session: `{st.session_state.thread_id[:8]}...`")
    # user_id দেখাও — debug এর জন্য
    st.caption(f"User: `{st.session_state.user_id[:8]}...`")

# =============================================================
# MAIN CHAT AREA
# =============================================================
st.title("🧠 RAG Memory Chat")
st.divider()

# =============================================================
# CHAT HISTORY DISPLAY
# =============================================================
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div style="display:flex;flex-direction:column;align-items:flex-end">'
            f'<div class="user-bubble">🧑 {msg["content"]}</div>'
            f'<div class="timestamp user-timestamp">{msg["time"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="display:flex;flex-direction:column;align-items:flex-start">'
            f'<div class="bot-bubble">🤖 {msg["content"]}</div>'
            f'<div class="timestamp">{msg["time"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

if st.session_state.is_thinking:
    st.markdown(
        '<div class="thinking">🤖 thinking...</div>',
        unsafe_allow_html=True
    )

# =============================================================
# INPUT BOX
# =============================================================
with st.form(key="chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="কিছু জিজ্ঞেস করো...",
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Send 🚀", use_container_width=True)

# =============================================================
# API CALL
# =============================================================
if submitted and user_input:

    now = datetime.now().strftime("%I:%M %p")

    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "time": now
    })

    st.session_state.is_thinking = True
    st.rerun()

# =============================================================
# THINKING STATE
# =============================================================
if st.session_state.is_thinking:

    try:
        response = requests.post(
            "http://api:8000/chat/",
            json={
                "thread_id": st.session_state.thread_id,
                # user_id → long-term memory এর জন্য
                # thread_id বদলালেও user_id same থাকে
                "user_id": st.session_state.user_id,
                "message": st.session_state.messages[-1]["content"]
            }
        )
        reply = response.json()["reply"]

    except Exception as e:
        reply = f"⚠️ Error: {str(e)}"

    now = datetime.now().strftime("%I:%M %p")

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "time": now
    })

    st.session_state.is_thinking = False
    st.rerun()