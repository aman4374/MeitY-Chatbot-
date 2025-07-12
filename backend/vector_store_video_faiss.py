import os
import json
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from backend.utils import ensure_directory, split_text_into_chunks

VIDEO_FAISS_DIR = "faiss_video"
VIDEO_METADATA_FILE = os.path.join(VIDEO_FAISS_DIR, "metadata.json")

ensure_directory(VIDEO_FAISS_DIR)

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
def load_faiss_video(embeddings):
    return FAISS.load_local("chroma_video_faiss", embeddings, allow_dangerous_deserialization=True)


    index_path = os.path.join(VIDEO_FAISS_DIR, "index")
    if os.path.exists(index_path):
        db = FAISS.load_local(index_path, embeddings)
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, embeddings)

    db.save_local(index_path)

    VIDEO_HASHES.add(video_hash)
    with open(VIDEO_METADATA_FILE, "w") as f:
        json.dump(list(VIDEO_HASHES), f)
