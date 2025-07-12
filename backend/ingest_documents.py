import os
from backend.utils import compute_md5, extract_text_from_file
from backend.vector_store_golden_faiss import persist_to_faiss_golden, is_already_ingested

def ingest_file(file_path: str) -> str:
    file_hash = compute_md5(file_path)

    # If already ingested and vectors exist, skip
    if is_already_ingested(file_hash) and os.path.exists("faiss_golden/index.faiss"):
        return "⚠️ File already ingested (duplicate detected)."

    # Extract text
    text = extract_text_from_file(file_path)
    if not text or len(text.strip()) < 20:
        return "❌ No readable content found in the file."

    # Persist to FAISS + store hash
    persist_to_faiss_golden(text, file_path, file_hash)
    return "✅ Document successfully embedded and saved."
