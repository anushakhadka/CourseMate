from pydantic import BaseModel
from typing import Optional

class Health(BaseModel):
    status: str

class IngestRequest(BaseModel):
    title: str
    text: str

class SearchRequest(BaseModel):
    query: str
    k: int = 8

class ChunkOut(BaseModel):
    id: int
    doc_id: int
    page: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    text: str
    score: float
