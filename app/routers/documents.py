# =============================================================
# FILE: app/routers/documents.py
# PURPOSE: document upload + Pinecone index endpoint
# CALLED BY: app/main.py

# FLOW:
#   Upload PDF/TXT
#     → text extract করো
#     → chunks বানাও
#     → OpenAI embeddings
#     → Pinecone এ store করো
# =============================================================

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from pinecone import Pinecone, ServerlessSpec
from app.database import save_document_record, get_user_documents
import tempfile
import os

router = APIRouter(prefix="/documents", tags=["documents"])

# =============================================================
# PINECONE SETUP — module load এ একবার চলে
# PURPOSE: Pinecone client বানানো + index create করা
# কেন এখানে: module load হলেই একবার চলে
# =============================================================
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ragmemory")
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# index না থাকলে বানাও
# dimensions=1536 → OpenAI text-embedding-ada-002 এর size
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
# কেন chunks: LLM এর context limit আছে, পুরো doc দেওয়া যায় না
# chunk_size=1000 → প্রতিটা chunk সর্বোচ্চ 1000 character
# chunk_overlap=200 → chunks এর মধ্যে 200 character overlap
#   কেন overlap: chunk boundary তে important info না হারায়
# =============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    message: str

# =============================================================
# ENDPOINT: POST /documents/upload
# PURPOSE: PDF বা TXT file upload করে Pinecone এ index করা
# PARAMETER: file → UploadFile (FastAPI automatic handle করে)
# RETURNS: UploadResponse
# =============================================================
@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
       # file extension check করো
    filename = file.filename
    if not filename.endswith((".pdf", ".txt")):
        raise HTTPException(
            status_code=400,
            detail="শুধু PDF বা TXT file support করা হয়"
        )
 # file টা temporarily disk এ save করো
    # কেন: PyPDFLoader/TextLoader এর file path দরকার
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(filename)[1]
    ) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # file type অনুযায়ী loader choose করো
        # PyPDFLoader → PDF থেকে text extract করে
        # TextLoader → TXT file পড়ে
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path, encoding="utf-8")
        # document load করো
        # OUTPUT: [Document(page_content="...", metadata={...})]

        documents = loader.load()
        # documents কে chunks এ ভাগ করো
        # OUTPUT: [Document, Document, ...] — ছোট ছোট pieces
        chunks = text_splitter.split_documents(documents)

          # documents কে chunks এ ভাগ করো
        # OUTPUT: [Document, Document, ...] — ছোট ছোট pieces

        # প্রতিটা chunk এ filename metadata যোগ করো
        # কেন: পরে জানা যাবে কোন file থেকে এসেছে
        for chunk in chunks:
            chunk.metadata["source_file"] = filename
            chunk.metadata["user_id"] = user_id

        # Pinecone এ namespace=user_id তে save করো
        user_vector_store = PineconeVectorStore(
            index_name=INDEX_NAME,
            embedding=embeddings,
            namespace=user_id
        )
        # chunks → embeddings → Pinecone এ store করো
        # PineconeVectorStore automatically embedding বানায়
        user_vector_store.add_documents(chunks)

        # PostgreSQL এ record রাখো
        # কেন: কোন file upload হয়েছে সেটা track করতে
        await save_document_record(user_id, filename, len(chunks))

        return UploadResponse(
            filename=filename,
            chunks_indexed=len(chunks),
            message=f"সফলভাবে {len(chunks)} টি chunk index করা হয়েছে"
        )

    finally:
        os.unlink(tmp_path)

# =============================================================
# ENDPOINT: GET /documents/list
# PURPOSE: একজন user এর সব uploaded documents list দেওয়া
# PARAMETER: user_id → query parameter
# RETURNS: {"documents": [{"filename": "...", "chunks": 10, ...}]}
# =============================================================
@router.get("/list")
async def list_documents(user_id: str):
    # PostgreSQL থেকে এই user এর documents আনো
    docs = await get_user_documents(user_id)
    return {
        "user_id": user_id,
        "documents": docs
    }