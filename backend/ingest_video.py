import os
import whisper
from backend.utils import compute_md5, ensure_directory, split_text_into_chunks
from backend.vector_store_video_faiss import persist_to_faiss_video, is_video_already_ingested


VIDEO_DIR = "video_upload"

def transcribe_video_to_text(video_path: str) -> str:
    """Transcribe audio from an MP4 video using Whisper."""
    try:
        model = whisper.load_model("base")  # or "medium"/"large" if you want higher accuracy
        result = model.transcribe(video_path)
        return result.get("text", "")
    except Exception as e:
        print(f"[Whisper Error] {e}")
        return ""

def ingest_video_file(video_path: str) -> str:
    ensure_directory(VIDEO_DIR)
    
    file_hash = compute_md5(video_path)
    if is_video_already_ingested(file_hash):
        return "⚠️ Video already ingested (duplicate detected)."

    text = transcribe_video_to_text(video_path)
    if not text or len(text.strip()) < 30:
        return "❌ No valid transcription found in video."

    persist_to_faiss_video(text, video_path, file_hash)
    return "✅ Video successfully transcribed and embedded."
