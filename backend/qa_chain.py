import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_together import ChatTogether
from langchain_core.messages import AIMessage
from backend.vector_store_golden_faiss import load_faiss_golden
from backend.vector_store_scraped_faiss import load_faiss_scraped
from backend.vector_store_video_faiss import load_faiss_video
from backend.web_search import search_tavily

def get_top_relevant_docs(vectorstore, query: str, k: int = 4, threshold: float = 0.6):
    """Return top documents below a certain similarity score."""
    results = vectorstore.similarity_search_with_score(query, k=k)
    filtered_docs = [doc for doc, score in results if score < threshold]
    return filtered_docs

def get_answer(query: str) -> str:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 1️⃣ Golden Data (Uploaded Files)
    try:
        golden_db = load_faiss_golden(embeddings)
        golden_docs = get_top_relevant_docs(golden_db, query)
        if golden_docs:
            context = "\n\n".join([doc.page_content for doc in golden_docs])
            return ask_llm(query, context, source="📄 Answer (from uploaded documents)")
    except Exception as e:
        print(f"[Golden Retrieval Error] {e}")

    # 2️⃣ Scraped Data
    try:
        scraped_db = load_faiss_scraped(embeddings)
        scraped_docs = get_top_relevant_docs(scraped_db, query)
        if scraped_docs:
            context = "\n\n".join([doc.page_content for doc in scraped_docs])
            return ask_llm(query, context, source="🌐 Answer (from scraped websites)")
    except Exception as e:
        print(f"[Scraped Retrieval Error] {e}")

    # 3️⃣ Video Data
    try:
        video_db = load_faiss_video(embeddings)
        video_docs = get_top_relevant_docs(video_db, query)
        if video_docs:
            context = "\n\n".join([doc.page_content for doc in video_docs])
            return ask_llm(query, context, source="🎬 Answer (from ingested videos)")
    except Exception as e:
        print(f"[Video Retrieval Error] {e}")

    # 4️⃣ Tavily Internet Search
    try:
        tavily_answer = search_tavily(query)
        return f"🌐 **Answer (via Internet Search):**\n{tavily_answer}"
    except Exception as e:
        return f"❌ Internet fallback failed: {e}"

def ask_llm(query: str, context: str, source: str) -> str:
    prompt = f"""You are an expert assistant. Use the context to answer concisely.

Context:
{context}

Question: {query}
Answer:"""

    llm = ChatTogether(model="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.4)
    response = llm.invoke(prompt)

    if isinstance(response, AIMessage):
        answer_text = response.content.strip()
    else:
        answer_text = str(response).strip()

    return f"{source}:\n{answer_text}"
