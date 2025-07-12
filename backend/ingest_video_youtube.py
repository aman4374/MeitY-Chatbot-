import os
import tempfile
from yt_dlp import YoutubeDL
import whisper
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

# Constants
VIDEO_DIR = "video_upload"
FAISS_DIR = "faiss_video"

# Whisper model (load once)
whisper_model = whisper.load_model("base")

def download_youtube_audio(youtube_url: str, output_path: str) -> str:
    os.makedirs(output_path, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_path, "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192"
        }],
        "quiet": True
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        filename = ydl.prepare_filename(info).replace(".webm", ".mp3").replace(".m4a", ".mp3")
        return filename

def transcribe_audio(path: str) -> str:
    result = whisper_model.transcribe(path)
    return result.get("text", "")

def ingest_youtube_video(youtube_url: str) -> str:
    try:
        print(f"📥 Downloading YouTube video: {youtube_url}")
        audio_path = download_youtube_audio(youtube_url, VIDEO_DIR)
        print("🎙️ Transcribing...")
        transcript = transcribe_audio(audio_path)

        if not transcript.strip():
            return "❌ Transcription failed or no speech detected."

        # Chunking
        chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 1000)]
        docs = [Document(page_content=chunk, metadata={"source": youtube_url}) for chunk in chunks]

        # Save to FAISS
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

        if os.path.exists(os.path.join(FAISS_DIR, "index.faiss")):
            vectordb = FAISS.load_local(FAISS_DIR, embeddings)
            vectordb.add_documents(docs)
        else:
            vectordb = FAISS.from_documents(docs, embeddings)

        vectordb.save_local(FAISS_DIR)
        return "✅ YouTube video successfully transcribed and indexed."

    except Exception as e:
        return f"❌ Failed to process YouTube video: {e}"
