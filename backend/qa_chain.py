import os
from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_together import ChatTogether
from langchain_core.messages import AIMessage
from backend.vector_store_golden_faiss import load_faiss_golden
from backend.vector_store_scraped_faiss import load_faiss_scraped
from backend.vector_store_video_faiss import load_faiss_video
from backend.web_search import search_tavily

# def get_top_relevant_docs(vectorstore, query: str, k: int = 4):
#     results = vectorstore.similarity_search_with_score(query, k=k)

#     print(f"Retrieved {len(results)} docs with scores:")
#     for doc, score in results:
#         print(f" - {score:.4f}")

#     # Just return top K (sorted)
#     return [doc for doc, _ in results]

def get_top_relevant_docs(vectorstore, query: str, k: int = 4, threshold: float = 1.5):
    """Return top documents above a similarity threshold (i.e., lower distance)."""
    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"Retrieved {len(results)} docs with scores:")
    for doc, score in results:
        print(f" - {score:.4f}")

    # ⛔ THIS LINE FILTERS OUT DOCUMENTS BASED ON THE THRESHOLD
    filtered_docs = [doc for doc, score in results if score < threshold]

    print(f"Filtered top docs (score < {threshold}): {len(filtered_docs)}")
    return filtered_docs



def get_answer(query: str) -> str:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # 1️⃣ Golden Data (Uploaded Files)
    try:
        golden_db = load_faiss_golden(embeddings)
        golden_docs = get_top_relevant_docs(golden_db, query)
        if golden_docs:
            print("✅ Answering from: 📄 Uploaded Documents")
            context = "\n\n".join([doc.page_content for doc in golden_docs])
            return ask_llm(query, context, source="📄 Answer (from uploaded documents)")
    except Exception as e:
        print(f"[Golden Retrieval Error] {e}")

    # 2️⃣ Scraped Data
    try:
        scraped_db = load_faiss_scraped(embeddings)
        scraped_docs = get_top_relevant_docs(scraped_db, query)
        if scraped_docs:
            print("✅ Answering from: 🌐 Scraped Websites")
            context = "\n\n".join([doc.page_content for doc in scraped_docs])
            return ask_llm(query, context, source="🌐 Answer (from scraped websites)")
    except Exception as e:
        print(f"[Scraped Retrieval Error] {e}")

    # 3️⃣ Video Data
    try:
        video_db = load_faiss_video(embeddings)
        video_docs = get_top_relevant_docs(video_db, query)
        if video_docs:
            print("✅ Answering from: 🎬 Video Transcripts")
            context = "\n\n".join([doc.page_content for doc in video_docs])
            return ask_llm(query, context, source="🎬 Answer (from ingested videos)")
    except Exception as e:
        print(f"[Video Retrieval Error] {e}")

    # 4️⃣ Tavily Internet Search
    try:
        print("🔍 Falling back to Internet (Tavily)...")
        tavily_answer = search_tavily(query)
        return f"🌐 **Answer(via Internet) :**\n{tavily_answer}"
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
