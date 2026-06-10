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
            background-color: #0084ff;
            padding: 10px 16px;
            border-radius: 18px 18px 4px 18px;
            margin: 4px 0;
            max-width: 65%;
            margin-left: auto;
            color: white;
            font-size: 14px;
            line-height: 1.5;
        }
        .bot-bubble {
            background-color: #2d2d2d;
            padding: 10px 16px;
            border-radius: 18px 18px 18px 4px;
            margin: 4px 0;
            max-width: 65%;
            color: #f0f0f0;
            font-size: 14px;
            line-height: 1.5;
        }
        .timestamp {
            font-size: 10px;
            color: #666;
            margin: 2px 4px;
        }
        .user-timestamp { text-align: right; }
        .thinking {
            background-color: #2d2d2d;
            padding: 10px 16px;
            border-radius: 18px 18px 18px 4px;
            max-width: 120px;
            color: #888;
            font-size: 14px;
            font-style: italic;
        }
        .doc-card {
            background-color: #1e1e2e;
            border: 1px solid #3d3d5c;
            padding: 10px 14px;
            border-radius: 10px;
            margin: 6px 0;
            font-size: 12px;
        }
        .doc-card b { color: #a0aec0; }
        .doc-meta { color: #666; font-size: 11px; margin-top: 3px; }
        .section-label {
            font-size: 11px;
            font-weight: 600;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 8px;
        }
        .upload-hint {
            font-size: 11px;
            color: #555;
            text-align: center;
            padding: 4px 0;
        }
        .identity-box {
            background-color: #1a1a2e;
            border: 1px solid #3d3d5c;
            border-radius: 8px;
            padding: 10px;
            margin: 6px 0;
            font-size: 11px;
            color: #888;
        }
    </style>
""", unsafe_allow_html=True)

# =============================================================
# SESSION STATE
# PURPOSE: browser refresh না হওয়া পর্যন্ত data রাখা
# FIELDS:
#   messages     → UI তে দেখানোর জন্য conversation history
#   thread_id    → এই session এর unique id — short-term memory
#   is_thinking  → LLM call চলছে কিনা
#   user_id      → long-term memory + document namespace
#                  browser session জুড়ে same থাকে
#                  thread_id বদলালেও user_id বদলায় না
#   pending_file → upload এর জন্য selected file
# =============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "is_thinking" not in st.session_state:
    st.session_state.is_thinking = False

if "user_id" not in st.session_state:
    # default_user → browser refresh করলেও same থাকবে
    # user নিজে YOUR IDENTITY section থেকে বদলাতে পারবে
    st.session_state.user_id = "default_user"

if "pending_file" not in st.session_state:
    st.session_state.pending_file = None

# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown("## 🧠 RAG Memory Chat")
    st.divider()

    # ==========================================================
    # SECTION 1: YOUR IDENTITY
    # PURPOSE: user_id set করা — long-term memory এর key
    # কেন এটা আগে: user_id set না করলে memory কাজ করবে না
    # ==========================================================
    st.markdown('<div class="section-label">👤 Your Identity</div>', unsafe_allow_html=True)

    # user_id দেখাও + edit করার option
    new_user_id = st.text_input(
        "Your User ID",
        value=st.session_state.user_id,
        placeholder="নিজের ID দাও...",
        help="এই ID দিয়ে তোমার memory আর documents track হয়। সব conversation এ same রাখো।",
        label_visibility="collapsed"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✅ Set ID", use_container_width=True):
            if new_user_id.strip():
                # user_id বদলালে messages clear করো
                # কেন: নতুন user এর পুরনো messages দেখা ঠিক না
                old_id = st.session_state.user_id
                st.session_state.user_id = new_user_id.strip()
                if old_id != new_user_id.strip():
                    st.session_state.messages = []
                st.rerun()
    with col_b:
        if st.button("🎲 Random", use_container_width=True, help="নতুন random ID বানাও"):
            st.session_state.user_id = str(uuid.uuid4())[:8]
            st.session_state.messages = []
            st.rerun()

    # current user_id দেখাও
    st.markdown(
        f'<div class="identity-box">🔑 Active ID: <b style="color:#7c8cf8">'
        f'{st.session_state.user_id}</b></div>',
        unsafe_allow_html=True
    )

    st.divider()

    # ==========================================================
    # SECTION 2: CONVERSATIONS
    # ==========================================================
    st.markdown('<div class="section-label">💬 Conversations</div>', unsafe_allow_html=True)

    # PostgreSQL থেকে conversation list আনো
    try:
        res = requests.get("http://api:8000/chat/conversations")
        conversations = res.json()["conversations"]
    except:
        conversations = []

    # current thread_id list এ না থাকলে যোগ করো
    if st.session_state.thread_id not in conversations:
        conversations = [st.session_state.thread_id] + conversations

    # Conversation dropdown
    if conversations:
        current_index = (
            conversations.index(st.session_state.thread_id)
            if st.session_state.thread_id in conversations
            else 0
        )
        selected = st.selectbox(
            "Active Conversation",
            options=conversations,
            index=current_index,
            format_func=lambda x: f"💬 {x[:12]}...",
            label_visibility="collapsed"
        )
        if selected != st.session_state.thread_id:
            st.session_state.thread_id = selected
            st.session_state.messages = []
            st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New", use_container_width=True, help="নতুন conversation শুরু করো"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Refresh", use_container_width=True, help="List update করো"):
            st.rerun()

    # নিজে ID দিয়ে load করার option
    with st.expander("🔗 Load by ID"):
        custom_id = st.text_input(
            "Conversation ID",
            placeholder="ID paste করো...",
            label_visibility="collapsed"
        )
        if st.button("Load", use_container_width=True):
            if custom_id.strip():
                st.session_state.thread_id = custom_id.strip()
                st.session_state.messages = []
                st.rerun()

    st.divider()

    # ==========================================================
    # SECTION 3: MY DOCUMENTS
    # PURPOSE: user এর uploaded documents দেখানো + নতুন upload
    # Documents user_id এর namespace এ Pinecone এ থাকে
    # ==========================================================
    st.markdown('<div class="section-label">📁 My Documents</div>', unsafe_allow_html=True)

    # এই user এর uploaded documents list আনো
    docs = []
    try:
        doc_res = requests.get(
            "http://api:8000/documents/list",
            params={"user_id": st.session_state.user_id}
        )
        doc_data = doc_res.json()
        docs = doc_data.get("documents", [])

        if docs:
            for doc in docs:
                uploaded_at = str(doc["uploaded_at"])[:10]
                st.markdown(
                    f'<div class="doc-card">'
                    f'📄 <b>{doc["filename"]}</b>'
                    f'<div class="doc-meta">{doc["chunks"]} chunks • {uploaded_at}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div style="color:#555;font-size:12px;text-align:center;padding:8px">'
                '📭 কোনো document নেই<br>নিচে upload করো'
                '</div>',
                unsafe_allow_html=True
            )
    except:
        st.caption("⚠️ Document list unavailable")

    # Upload section
    st.markdown("&nbsp;", unsafe_allow_html=True)
    st.markdown(
        '<div class="upload-hint">PDF, TXT বা DOCX upload করো</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload",
        type=["pdf", "txt", "docx"],
        label_visibility="collapsed",
        key="doc_uploader"
    )

    # file select হলে session_state এ রাখো
    # কেন session_state: file_uploader আর button একই
    # rerun এ কাজ করে না — session_state দিয়ে ধরে রাখতে হয়
    if uploaded_file is not None:
        st.session_state.pending_file = uploaded_file

    # pending file থাকলে Index button দেখাও
    if st.session_state.pending_file is not None:
        fname = st.session_state.pending_file.name
        st.markdown(
            f'<div style="font-size:11px;color:#888;margin:4px 0">📄 {fname}</div>',
            unsafe_allow_html=True
        )
        if st.button("📤 Index Document", use_container_width=True, type="primary"):
            file_to_upload = st.session_state.pending_file
            with st.spinner(f"'{file_to_upload.name}' index হচ্ছে..."):
                try:
                    upload_res = requests.post(
                        "http://api:8000/documents/upload",
                        files={
                            "file": (
                                file_to_upload.name,
                                file_to_upload.getvalue(),
                                file_to_upload.type or "application/octet-stream"
                            )
                        },
                        # user_id Form field হিসেবে পাঠাও
                        # কেন Form: file upload এর সাথে extra data পাঠাতে Form লাগে
                        data={"user_id": st.session_state.user_id}
                    )

                    if upload_res.status_code == 200:
                        res_data = upload_res.json()
                        st.success(
                            f"✅ **{res_data['filename']}**\n\n"
                            f"{res_data['chunks_indexed']} chunks indexed!"
                        )
                        st.session_state.pending_file = None
                        st.rerun()
                    else:
                        st.error(f"❌ {upload_res.json().get('detail', 'Upload failed')}")

                except Exception as e:
                    st.error(f"⚠️ {str(e)}")

    st.divider()

    # ==========================================================
    # SECTION 4: SESSION INFO
    # ==========================================================
    st.markdown('<div class="section-label">ℹ️ Session Info</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Messages", len(st.session_state.messages))
    with col2:
        st.metric("Documents", len(docs))

    st.caption(f"🔑 Thread: `{st.session_state.thread_id[:8]}...`")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# =============================================================
# MAIN CHAT AREA
# =============================================================
st.title("🧠 RAG Memory Chat")

st.caption(
    f"Active session: `{st.session_state.thread_id[:8]}...` • "
    f"User: `{st.session_state.user_id}` • "
    f"{'📚 Documents indexed' if docs else '📭 No documents'}"
)
st.divider()

# =============================================================
# CHAT HISTORY DISPLAY
# =============================================================
if not st.session_state.messages:
    st.markdown(
        '<div style="text-align:center;color:#444;padding:60px 0">'
        '<div style="font-size:48px">🧠</div>'
        '<div style="font-size:16px;margin-top:12px">কথা শুরু করো!</div>'
        '<div style="font-size:12px;color:#555;margin-top:6px">'
        'Document upload করলে সেটা থেকেও উত্তর দিতে পারবো।'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div style="display:flex;flex-direction:column;align-items:flex-end;margin:8px 0">'
            f'<div class="user-bubble">{msg["content"]}</div>'
            f'<div class="timestamp user-timestamp">{msg["time"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div style="display:flex;flex-direction:column;align-items:flex-start;margin:8px 0">'
            f'<div class="bot-bubble">{msg["content"]}</div>'
            f'<div class="timestamp">{msg["time"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

# Typing indicator
if st.session_state.is_thinking:
    st.markdown(
        '<div style="margin:8px 0">'
        '<div class="thinking">🤖 typing...</div>'
        '</div>',
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
            placeholder="এখানে লেখো... (Enter চাপো)",
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
# THINKING STATE — API call করো
# কেন আলাদা block: rerun() এর পরে is_thinking True থাকলে
# এই block চলবে — thinking indicator দেখাবে তারপর API call
# =============================================================
if st.session_state.is_thinking:
    try:
        response = requests.post(
            "http://api:8000/chat/",
            json={
                "thread_id": st.session_state.thread_id,
                # user_id → long-term memory + Pinecone namespace
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