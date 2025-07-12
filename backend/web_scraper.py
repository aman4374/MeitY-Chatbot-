import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from backend.utils import compute_md5
from backend.vector_store_scraped_faiss import is_scraped_already, persist_to_faiss_scraped

def scrape_visible_text(url: str) -> str:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove script/style/noscript content
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        visible_text = soup.get_text(separator="\n")
        lines = [line.strip() for line in visible_text.splitlines() if line.strip()]
        return "\n".join(lines)

    except Exception as e:
        return f"❌ Failed to scrape URL: {e}"

def scrape_and_ingest(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return "❌ Invalid URL"

    text = scrape_visible_text(url)
    if text.startswith("❌"):
        return text

    url_hash = compute_md5(text)
    if is_scraped_already(url_hash):
        return "⚠️ URL already ingested (duplicate detected)."

    persist_to_faiss_scraped(text, url, url_hash)
    return "✅ Website scraped and embedded successfully."
