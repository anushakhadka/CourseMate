from sqlalchemy import Column, Integer, String, Text
from pgvector.sqlalchemy import Vector
from .db import Base

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    title = Column(String(512), nullable=False)
    source = Column(String(1024), nullable=True)

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True)
    doc_id = Column(Integer, nullable=False)
    page = Column(Integer, nullable=True)
    line_start = Column(Integer, nullable=True)
    line_end = Column(Integer, nullable=True)
    text = Column(Text, nullable=False)
    embedding = Column(Vector(1536))  # fits text-embedding-3-* dims
