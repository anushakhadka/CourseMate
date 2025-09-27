from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.types import Integer
from sqlalchemy import text, bindparam
from sqlalchemy.orm import Session
from typing import List

from pgvector.sqlalchemy import Vector  # for binding vector params
from .search import ensure_index, rrf
from .db import Base, engine, SessionLocal, init_pgvector
from .models import Document, Chunk
from .schemas import Health, IngestRequest, SearchRequest, ChunkOut
from .embeddings import embed
from .ingest_pdf import extract_chunks_from_pdf


app = FastAPI(title="CourseMate Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- DB session dependency ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# --- Startup: init pgvector extension + tables ---
@app.on_event("startup")
def on_startup():
    init_pgvector()  # creates extension and registers pgvector adapter
    Base.metadata.create_all(bind=engine)
    ensure_index()



# --- Health ---
@app.get("/health", response_model=Health)
async def health():
    return {"status": "ok"}


# --- Ingest raw text (quick test path) ---
@app.post("/ingest", response_model=Health)
async def ingest(payload: IngestRequest, db: Session = Depends(get_db)):
    # 1) Create document
    doc = Document(title=payload.title, source=None)
    db.add(doc)
    db.flush()  # get doc.id

    # 2) Naive paragraph chunking
    raw_chunks = [c.strip() for c in payload.text.split("\n\n") if c.strip()]
    if not raw_chunks:
        raise HTTPException(status_code=400, detail="No non-empty paragraphs found")

    # 3) Embed and store
    vectors = await embed(raw_chunks)
    for text_chunk, vec in zip(raw_chunks, vectors):
        db.add(Chunk(doc_id=doc.id, text=text_chunk, embedding=vec))

    db.commit()
    return {"status": f"ingested {len(raw_chunks)} chunks into doc {doc.id}"}


# --- Ingest PDF (layout-aware; records page/line spans) ---
@app.post("/ingest_pdf", response_model=Health)
async def ingest_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a .pdf file")

    pdf_bytes = await file.read()
    parsed = extract_chunks_from_pdf(pdf_bytes)  # [(page, line_start, line_end, text), ...]

    if not parsed:
        raise HTTPException(
            status_code=400,
            detail="No text extracted from PDF. (Is it a scanned image? Try a selectable-text PDF.)",
        )

    # Create doc
    doc = Document(title=file.filename, source=file.filename)
    db.add(doc)
    db.flush()

    # Embed and store with spans
    texts = [t for (_, _, _, t) in parsed]
    vectors = await embed(texts)

    for (page, ls, le, text_chunk), vec in zip(parsed, vectors):
        db.add(
            Chunk(
                doc_id=doc.id,
                page=page,
                line_start=ls,
                line_end=le,
                text=text_chunk,
                embedding=vec,
            )
        )

    db.commit()
        # index into Meilisearch (BM25)
    docs_for_meili = []
    # re-query the chunks for this doc (cheap and safe)
    rows = db.execute(text("SELECT id, doc_id, page, line_start, line_end, text FROM chunks WHERE doc_id=:d"),
                      {"d": doc.id}).mappings().all()
    for row in rows:
        docs_for_meili.append({
            "id": row["id"],               # primary key in Meili
            "doc_id": row["doc_id"],
            "page": row["page"],
            "line_start": row["line_start"],
            "line_end": row["line_end"],
            "text": row["text"],
            "title": payload.title,
        })
    ensure_index().add_documents(docs_for_meili)

    return {"status": f"ingested {len(parsed)} chunks from {file.filename} into doc {doc.id}"}


# --- Semantic search (pgvector; cosine distance) ---
@app.post("/search", response_model=List[ChunkOut])
async def search(req: SearchRequest, db: Session = Depends(get_db)):
    # 1) Embed query text
    qvec = (await embed([req.query]))[0]

    # 2) Vector similarity search (cosine distance). Bind qvec as a pgvector explicitly.
    sql = (
        text(
            """
            SELECT id, doc_id, page, line_start, line_end, text,
                   1 - (embedding <=> :qvec) AS score
            FROM chunks
            ORDER BY embedding <=> :qvec
            LIMIT :k
            """
        )
        .bindparams(
         bindparam("qvec", type_=Vector(1536)),
         bindparam("k", type_=Integer()),
        )

    )

    rows = db.execute(sql, {"qvec": qvec, "k": req.k}).mappings().all()
    return [ChunkOut(**row) for row in rows]
@app.post("/search_hybrid", response_model=List[ChunkOut])
async def search_hybrid(req: SearchRequest, db: Session = Depends(get_db)):
    # 1) Vector side (same as /search, but only ids)
    qvec = (await embed([req.query]))[0]
    sql_vec = (
        text(
            """
            SELECT id
            FROM chunks
            ORDER BY embedding <=> :qvec
            LIMIT :k
            """
        )
        .bindparams(
            bindparam("qvec", type_=Vector(1536)),
            bindparam("k", type_=Integer()),
        )
    )
    vec_ids = [r["id"] for r in db.execute(sql_vec, {"qvec": qvec, "k": req.k}).mappings().all()]

    # 2) BM25 side (Meilisearch)
    idx = ensure_index()
    bm25_hits = idx.search(req.query, {"limit": req.k})
    bm25_ids = [hit["id"] for hit in bm25_hits.get("hits", [])]

    # 3) Fuse with RRF
    fused_ids = rrf(bm25_ids, vec_ids, k=60)[: req.k]
    if not fused_ids:
        return []

    # 4) Fetch rows and compute a simple composite score for display
    sql_fetch = text(
        """
        SELECT id, doc_id, page, line_start, line_end, text,
               1 - (embedding <=> :qvec) AS score_vec
        FROM chunks
        WHERE id = ANY(:ids)
        """
    ).bindparams(
        bindparam("qvec", type_=Vector(1536)),
        bindparam("ids", expanding=True),
    )
    rows = db.execute(sql_fetch, {"qvec": qvec, "ids": fused_ids}).mappings().all()
    # map id -> row
    row_by_id = {r["id"]: r for r in rows}
    # build output in fused order; expose vec score as 'score'
    out: List[ChunkOut] = []
    for cid in fused_ids:
        r = row_by_id.get(cid)
        if r:
            out.append(ChunkOut(
                id=r["id"], doc_id=r["doc_id"], page=r["page"],
                line_start=r["line_start"], line_end=r["line_end"],
                text=r["text"], score=float(r["score_vec"])
            ))
    return out
