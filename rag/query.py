"""
RAG Query Pipeline
Semantic search over ChromaDB + Groq generate answer.
"""

import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

CHROMA_DIR  = "data/chroma_db"
COLLECTION  = "thongtu_ppe"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K       = 4

SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn an toàn lao động tại Việt Nam.
Trả lời câu hỏi dựa trên các quy định được cung cấp bên dưới.
Nếu thông tin không có trong tài liệu, hãy nói rõ là không tìm thấy quy định cụ thể.
Trích dẫn điều khoản hoặc ngành nghề cụ thể khi có thể.
Trả lời bằng tiếng Việt, ngắn gọn và rõ ràng."""


_model:      SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_groq:       Groq | None = None


def _init():
    global _model, _collection, _groq
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    if _collection is None:
        client      = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION)
    if _groq is None:
        _groq = Groq(api_key=os.environ["GROQ_API_KEY"])


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return top-k relevant chunks for a query."""
    _init()
    query_vec = _model.encode([query]).tolist()
    results   = _collection.query(
        query_embeddings=query_vec,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":     doc,
            "source":   meta.get("source"),
            "section":  meta.get("section", ""),
            "dieu":     meta.get("dieu", ""),
            "page":     meta.get("page"),
            "score":    round(1 - dist, 4),   # cosine similarity
        })
    return chunks


def answer(query: str, k: int = TOP_K) -> dict:
    """
    Full RAG pipeline: retrieve → augment → generate.
    Returns answer text + source chunks.
    """
    _init()
    chunks = retrieve(query, k)

    context_blocks = []
    for i, c in enumerate(chunks, 1):
        label = f"Điều {c['dieu']}" if c["source"] == "dieu_khoan" else c["section"]
        context_blocks.append(f"[Nguồn {i} - {label} - Trang {c['page']}]\n{c['text']}")

    context = "\n\n".join(context_blocks)

    prompt = f"""Các quy định liên quan từ Thông tư 25/2022/TT-BLĐTBXH:

{context}

---
Câu hỏi: {query}"""

    response = _groq.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.2,
        max_tokens=600,
    )

    return {
        "question": query,
        "answer":   response.choices[0].message.content.strip(),
        "sources":  chunks,
    }


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    test_queries = [
        "Công nhân xây dựng lắp đặt kết cấu thép cần trang bị PPE gì?",
        "Điều kiện để được cấp phương tiện bảo vệ cá nhân là gì?",
        "Ai chịu trách nhiệm kiểm tra chất lượng PPE?",
    ]

    for q in test_queries:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        result = answer(q)
        print(f"A: {result['answer']}")
        print(f"\nSources:")
        for s in result["sources"]:
            print(f"  - [{s['source']}] page {s['page']} | score {s['score']}")