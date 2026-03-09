"""
RAG Ingest Pipeline - Thong tu 25/2022/TT-BLDTBXH
Chunks document, embeds, stores in ChromaDB.
Run once to build the index.
"""

import re
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

PDF_PATH    = "data/25_2022_TT-BLDTBXH_551396.pdf"
CHROMA_DIR  = "data/chroma_db"
COLLECTION  = "thongtu_ppe"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Dieu khoan pages (0-indexed)
DIEU_KHOAN_PAGES = range(0, 4)

# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_dieu_khoan(pdf) -> list[dict]:
    chunks = []
    full_text = ""
    page_map = {}  # track which page each character came from

    # Build full text with page boundaries marked
    for page_idx in DIEU_KHOAN_PAGES:
        text = pdf.pages[page_idx].extract_text() or ""
        page_map[len(full_text)] = page_idx + 1  # char offset -> page number
        full_text += text + "\n"

    pattern = r"(Điều \d+\.[^\n]*\n)"
    parts = re.split(pattern, full_text)

    # Track char offset to find correct page
    offset = 0
    i = 1
    while i < len(parts) - 1:
        header = parts[i].strip()
        body   = parts[i + 1].strip() if i + 1 < len(parts) else ""
        chunk_text = f"{header}\n{body}"

        if len(chunk_text) > 50:
            dieu_num   = re.search(r"Điều (\d+)", header)
            dieu_title = re.search(r"Điều \d+\.\s*(.+)", header)

            # Find page number from offset
            chunk_offset = full_text.find(parts[i], offset)
            page = 1
            for char_pos, pg in sorted(page_map.items()):
                if char_pos <= chunk_offset:
                    page = pg

            chunks.append({
                "text":        chunk_text,
                "source":      "dieu_khoan",
                "dieu":        dieu_num.group(1) if dieu_num else "?",
                "dieu_title":  dieu_title.group(1).strip() if dieu_title else "",
                "page":        str(page),
            })
            offset = chunk_offset

        i += 2

    print(f"[ingest] Dieu khoan: {len(chunks)} chunks")
    return chunks

def extract_phu_luc_tables(pdf) -> list[dict]:
    """
    Extract Phu luc I (page 5-229) using structural table parsing.
    Bypasses text mixing issues inherent to Regex text extraction.
    """
    chunks = []
    for page_idx in range(4, len(pdf.pages)):
        page = pdf.pages[page_idx]
        tables = page.extract_tables()
        
        if not tables:
            continue
            
        for table in tables:
            for row in table:
                if not row or len(row) < 4:
                    continue
                    
                if not row[1] or "Nghề, công việc" in row[1]:
                    continue
                    
                job_desc  = str(row[1]).replace('\n', ' ').strip()
                condition = str(row[2]).replace('\n', ' ').strip()
                ppe       = str(row[3]).replace('\n', ' ').strip()
                
                if not job_desc or not ppe:
                    continue
                    
                structured_text = (
                    f"Nghề, công việc: {job_desc}\n"
                    f"Điều kiện lao động: {condition}\n"
                    f"Phương tiện bảo vệ cá nhân bắt buộc: {ppe}"
                )
                
                chunks.append({
                    "text":    structured_text,
                    "source":  "phu_luc_table",
                    "page":    str(page_idx + 1)
                })

    print(f"[ingest] Phu luc (Tables): {len(chunks)} chunks")
    return chunks

# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def build_index(chunks: list[dict], model: SentenceTransformer) -> None:
    client     = chromadb.PersistentClient(path=CHROMA_DIR)

    # Reset collection if exists
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass    

    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts     = [c["text"]   for c in chunks]
    metadatas = [{k: v for k, v in c.items() if k != "text"} for c in chunks]
    ids       = [f"chunk_{i}" for i in range(len(chunks))]

    print(f"[ingest] Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    print(f"[ingest] Done. {len(texts)} chunks indexed to {CHROMA_DIR}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Path("data").mkdir(exist_ok=True)

    print("[ingest] Loading embedding model...")
    model = SentenceTransformer(EMBED_MODEL)

    print("[ingest] Reading PDF...")
    with pdfplumber.open(PDF_PATH) as pdf:
        chunks  = extract_dieu_khoan(pdf)
        chunks += extract_phu_luc_tables(pdf)

    print(f"[ingest] Total chunks: {len(chunks)}")
    build_index(chunks, model)

if __name__ == "__main__":
    main()