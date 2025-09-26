from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List

from .db import Base, engine, SessionLocal, init_pgvector
from .models import Document, Chunk
from .schemas import Health, IngestRequest, SearchRequest, ChunkOut
from .embeddings import embed

app = FastAPI(title="CourseMate Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.on_event("startup")
def on_startup():
    init_pgvector()
    Base.metadata.create_all(bind=engine)

@app.get("/health", response_model=Health)
async def health():
    return {"status": "ok"}

@app.post("/ingest", response_model=Health)
async def ingest(payload: IngestRequest, db: Session = Depends(get_db)):
    doc = Document(title=payload.title, source=None)
    db.add(doc)
    db.flush()

    raw_chunks = [c.strip() for c in payload.text.split("\n\n") if c.strip()]
    vectors = await embed(raw_chunks)

    for text_chunk, vec in zip(raw_chunks, vectors):
        db.add(Chunk(doc_id=doc.id, text=text_chunk, embedding=vec))

    db.commit()
    return {"status": f"ingested {len(raw_chunks)} chunks into doc {doc.id}"}

@app.post("/search", response_model=List[ChunkOut])
async def search(req: SearchRequest, db: Session = Depends(get_db)):
    qvec = (await embed([req.query]))[0]
    sql = text(
        """
        SELECT id, doc_id, page, line_start, line_end, text,
               1 - (embedding <=> :qvec) AS score
        FROM chunks
        ORDER BY embedding <=> :qvec
        LIMIT :k
        """
    )
    rows = db.execute(sql, {"qvec": qvec, "k": req.k}).mappings().all()
    return [ChunkOut(**row) for row in rows]
