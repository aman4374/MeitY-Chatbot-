import os
import json
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.docstore.document import Document
from backend.utils import compute_md5, split_text_into_chunks, ensure_directory
from backend.utils import compute_text_md5

PERSIST_DIR = "chroma_scraped_faiss"
METADATA_FILE = os.path.join(PERSIST_DIR, "scraped_metadata.json")

# Ensure the directory exists
ensure_directory(PERSIST_DIR)

# Load already scraped hashes
if os.path.exists(METADATA_FILE):
    with open(METADATA_FILE, "r") as f:
        SCRAPED_HASHES = set(json.load(f))
else:
    SCRAPED_HASHES = set()
    with open(METADATA_FILE, "w") as f:
        json.dump([], f)

def is_scraped_already(text: str) -> bool:
    """Compute hash of scraped text to avoid reprocessing."""
    content_hash = compute_text_md5(text)
    return content_hash in SCRAPED_HASHES

def persist_to_faiss_scraped(text: str, source_url: str, url_hash: str):
    """Embed and persist scraped text content into FAISS."""
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    chunks = split_text_into_chunks(text)
    docs = [Document(page_content=chunk, metadata={"source": source_url, "hash": url_hash}) for chunk in chunks]

    # Load existing DB or create new
    if os.path.exists(os.path.join(PERSIST_DIR, "index.faiss")):
        db = FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, embeddings)

    db.save_local(PERSIST_DIR)

    # Save hash to avoid re-scraping
    SCRAPED_HASHES.add(url_hash)
    with open(METADATA_FILE, "w") as f:
        json.dump(list(SCRAPED_HASHES), f)

def load_faiss_scraped(embeddings):
    index_path = os.path.join(PERSIST_DIR, "index.faiss")
    if not os.path.exists(index_path):
        raise RuntimeError("❌ FAISS index for scraped data not found. Scrape and ingest at least one URL.")

    return FAISS.load_local(PERSIST_DIR, embeddings, allow_dangerous_deserialization=True)
