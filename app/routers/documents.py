# =============================================================
# FILE: app/routers/documents.py
# PURPOSE: document upload + Pinecone index endpoint
# CALLED BY: app/main.py
# FLOW:
#   Upload PDF/TXT/DOCX
#     → text extract করো
#     → chunks বানাও
#     → OpenAI embeddings
#     → Pinecone এ namespace=user_id তে store করো
#     → PostgreSQL এ record রাখো
# =============================================================

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from pinecone import Pinecone, ServerlessSpec
from app.database import save_document_record, get_user_documents
import tempfile
import os

router = APIRouter(prefix="/documents", tags=["documents"])

# =============================================================
# PINECONE SETUP — module load এ একবার চলে
# PURPOSE: Pinecone client বানানো + index create করা
# =============================================================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ragmemory")

# OpenAI embeddings — text → vector বানায়
# dimensions=1536 → text-embedding-ada-002 এর size
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# index না থাকলে বানাও
# metric="cosine" → similarity search এর জন্য
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# =============================================================
# TEXT SPLITTER
# PURPOSE: বড় document কে ছোট chunks এ ভাগ করা
# chunk_size=1000  → প্রতিটা chunk সর্বোচ্চ 1000 character
# chunk_overlap=200 → boundary তে important info না হারায়
# =============================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# =============================================================
# RESPONSE MODEL
# =============================================================
class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str

# =============================================================
# ENDPOINT: POST /documents/upload
# PURPOSE: PDF, TXT বা DOCX upload করে Pinecone + PostgreSQL এ save
# PARAMETER:
#   file    → UploadFile
#   user_id → Form field — কোন user এর namespace এ save করবো
# RETURNS: UploadResponse
# =============================================================
@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    filename = file.filename

    # file extension check করো
    if not filename.endswith((".pdf", ".txt", ".docx")):
        raise HTTPException(
            status_code=400,
            detail="শুধু PDF, TXT বা DOCX file support করা হয়"
        )

    # file টা temporarily disk এ save করো
    # কেন: loader গুলোর file path দরকার
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(filename)[1]
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # file type অনুযায়ী loader choose করো
        # PyPDFLoader    → PDF থেকে text extract করে
        # Docx2txtLoader → DOCX থেকে text extract করে
        # TextLoader     → TXT file পড়ে
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")

        # document load করো
        # OUTPUT: [Document(page_content="...", metadata={...})]
        documents = loader.load()

        # documents কে chunks এ ভাগ করো
        # OUTPUT: [Document, Document, ...] — ছোট ছোট pieces
        chunks = text_splitter.split_documents(documents)

        # প্রতিটা chunk এ metadata যোগ করো
        # কেন: পরে জানা যাবে কোন file থেকে এসেছে
        for chunk in chunks:
            chunk.metadata["source_file"] = filename
            chunk.metadata["user_id"] = user_id

        # Pinecone এ namespace=user_id তে save করো
        # কেন namespace: প্রতিটা user এর document আলাদা থাকবে
        user_vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=user_id
        )
        user_vector_store.add_documents(chunks)

        # PostgreSQL এ record রাখো
        # কেন: কোন file upload হয়েছে সেটা sidebar এ দেখাবে
        await save_document_record(user_id, filename, len(chunks))

        return UploadResponse(
            filename=filename,
            chunks_indexed=len(chunks),
            message=f"সফলভাবে {len(chunks)} টি chunk index করা হয়েছে"
        )

    finally:
        # temporary file delete করো
        os.unlink(tmp_path)

# =============================================================
# ENDPOINT: GET /documents/list
# PURPOSE: একজন user এর সব uploaded documents list দেওয়া
# PARAMETER: user_id → query parameter
# RETURNS: {"user_id": "...", "documents": [...]}
# =============================================================
@router.get("/list")
async def list_documents(user_id: str):
    # PostgreSQL থেকে এই user এর documents আনো
    docs = await get_user_documents(user_id)
    return {
        "user_id": user_id,
        "documents": docs
    }