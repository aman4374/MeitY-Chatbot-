import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from backend.utils import ensure_directory, split_text_into_chunks

# Folder paths
VIDEO_FAISS_DIR = "faiss_video"
VIDEO_METADATA_FILE = os.path.join(VIDEO_FAISS_DIR, "metadata.json")

# Ensure the directory exists
ensure_directory(VIDEO_FAISS_DIR)

# Load or initialize ingested video hashes
if os.path.exists(VIDEO_METADATA_FILE):
    with open(VIDEO_METADATA_FILE, "r") as f:
        VIDEO_HASHES = set(json.load(f))
else:
    VIDEO_HASHES = set()
    with open(VIDEO_METADATA_FILE, "w") as f:
        json.dump([], f)

def is_video_already_ingested(video_hash: str) -> bool:
    return video_hash in VIDEO_HASHES

def persist_to_faiss_video(text: str, source: str, video_hash: str):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    chunks = split_text_into_chunks(text)
    docs = [
        Document(page_content=chunk, metadata={"source": source, "hash": video_hash})
        for chunk in chunks
    ]

    index_path = os.path.join(VIDEO_FAISS_DIR)

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, embeddings)

    db.save_local(index_path)

    # Update ingested hashes
    VIDEO_HASHES.add(video_hash)
    with open(VIDEO_METADATA_FILE, "w") as f:
        json.dump(list(VIDEO_HASHES), f)

def load_faiss_video(embeddings):
    index_path = VIDEO_FAISS_DIR
    if not os.path.exists(os.path.join(index_path, "index.faiss")):
        raise RuntimeError("❌ FAISS video index not found. Please ingest a video first.")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
