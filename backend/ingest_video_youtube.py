import os
import warnings
from yt_dlp import YoutubeDL
import whisper
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings

# Suppress the specific FutureWarning from PyTorch about pickle deserialization
warnings.filterwarnings(
    "ignore",
    message="You are using `torch.load` with `weights_only=False`",
    category=FutureWarning
)

# --- Define base path for persistent data ---
PERSISTENT_DIR = os.environ.get("PERSISTENT_STORAGE_PATH", "persistent_storage")
VIDEO_DIR = os.path.join(PERSISTENT_DIR, "video_upload")
FAISS_DIR = os.path.join(PERSISTENT_DIR, "faiss_video")
# --- NEW: Define the path for the cookies file ---
COOKIE_FILE_PATH = os.path.join(PERSISTENT_DIR, 'cookies.txt')


# Load the Whisper model once to be reused
whisper_model = whisper.load_model("base")

def download_youtube_audio(youtube_url: str, output_path: str) -> str:
    """Downloads audio from a YouTube URL and returns the file path."""
    os.makedirs(output_path, exist_ok=True)

    # --- MODIFIED: Added cookiefile option ---
    # This tells yt-dlp to use the cookies file for authentication,
    # which is necessary to bypass "confirm you're not a bot" checks.
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_path, "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192"
        }],
        "quiet": True,
    }

    # Only add the cookiefile option if the file actually exists
    if os.path.exists(COOKIE_FILE_PATH):
        print(f"🍪 Using cookies from {COOKIE_FILE_PATH}")
        ydl_opts['cookiefile'] = COOKIE_FILE_PATH
    else:
        print(f"⚠️ Cookie file not found at {COOKIE_FILE_PATH}. Proceeding without authentication.")
    # --- END MODIFICATION ---

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        # The filename is determined after download and post-processing
        base, _ = os.path.splitext(ydl.prepare_filename(info))
        filename = base + ".mp3"
        return filename


def transcribe_audio(path: str) -> str:
    """Transcribes an audio file and returns the text."""
    result = whisper_model.transcribe(path)
    return result.get("text", "")

def ingest_youtube_video(youtube_url: str) -> str:
    """Full pipeline to download, transcribe, and index a YouTube video."""
    audio_path = "" # Initialize to ensure it exists for the finally block
    try:
        print(f"📥 Downloading YouTube video: {youtube_url}")
        audio_path = download_youtube_audio(youtube_url, VIDEO_DIR)

        print("🎙️ Transcribing...")
        transcript = transcribe_audio(audio_path)

        if not transcript.strip():
            return "❌ Transcription failed or no speech detected."

        # Chunk the transcript and create Document objects
        chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 1000)]
        docs = [Document(page_content=chunk, metadata={"source": youtube_url}) for chunk in chunks]

        # Create embeddings and save to FAISS vector store
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        os.makedirs(FAISS_DIR, exist_ok=True)
        faiss_index_file = os.path.join(FAISS_DIR, "index.faiss")

        if os.path.exists(faiss_index_file):
            print(f"🔄 Loading and updating existing FAISS index from {FAISS_DIR}")
            vectordb = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
            vectordb.add_documents(docs)
        else:
            print(f"✨ Creating new FAISS index in {FAISS_DIR}")
            vectordb = FAISS.from_documents(docs, embeddings)

        vectordb.save_local(FAISS_DIR)
        return "✅ YouTube video successfully transcribed and indexed."

    except Exception as e:
        print(f"An error occurred during YouTube video ingestion: {e}")
        return f"❌ Failed to process YouTube video: {e}"

    finally:
        # Clean up the downloaded audio file after processing
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)