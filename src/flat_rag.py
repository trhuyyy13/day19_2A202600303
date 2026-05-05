from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import TOP_K_CHUNKS
from src.openai_utils import chat_completion


ANSWER_PROMPT = """Answer the question using only the context below.
If the context is insufficient, say that the context is insufficient.

Question:
{query}

Context:
{retrieved_context}
"""


def retrieve_chunks(query: str, chunks: list[dict], top_k: int = TOP_K_CHUNKS) -> list[dict]:
    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    top_indices = scores.argsort()[::-1][:top_k]
    return [chunks[index] for index in top_indices if scores[index] > 0]


def flat_rag_answer(query: str, chunks: list[dict], top_k: int = TOP_K_CHUNKS) -> dict:
    retrieved = retrieve_chunks(query, chunks, top_k=top_k)
    retrieved_context = "\n\n".join(
        f"Source: {chunk['title']} ({chunk['chunk_id']})\n{chunk['text']}" for chunk in retrieved
    )
    if not retrieved_context:
        answer = "The context is insufficient."
    else:
        answer = chat_completion(
            ANSWER_PROMPT.format(query=query, retrieved_context=retrieved_context)
        )

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": [chunk["chunk_id"] for chunk in retrieved],
        "method": "Flat RAG",
    }
