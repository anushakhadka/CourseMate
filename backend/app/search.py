from __future__ import annotations
from typing import List, Dict, Tuple
import os, math
import meilisearch

MEILI_HOST = os.getenv("MEILI_HOST", "http://meilisearch:7700")
MEILI_MASTER_KEY = os.getenv("MEILI_MASTER_KEY", "changeme")
INDEX_NAME = "chunks"

def meili() -> meilisearch.Client:
    return meilisearch.Client(MELI_HOST := MEILI_HOST, MEILI_MASTER_KEY)

def ensure_index() -> meilisearch.Index:
    client = meili()
    try:
        idx = client.get_index(INDEX_NAME)
    except meilisearch.errors.MeilisearchApiError:
        idx = client.create_index(INDEX_NAME, {"primaryKey": "id"})
        # configure searchable/filterable fields
        idx.update_searchable_attributes(["text", "title"])
        idx.update_filterable_attributes(["doc_id", "page"])
    return idx

def rrf(bm25_ranked: List[int], vec_ranked: List[int], k: int = 60) -> List[int]:
    """
    Simple Reciprocal Rank Fusion: score(d) = Σ 1 / (k + rank_d(list))
    Inputs are lists of ids in ranked order (best→worst).
    Returns a list of ids by fused score.
    """
    score: Dict[int, float] = {}
    for lst in [bm25_ranked, vec_ranked]:
        for r, did in enumerate(lst, start=1):
            score[did] = score.get(did, 0.0) + 1.0 / (k + r)
    return [did for did, _ in sorted(score.items(), key=lambda x: x[1], reverse=True)]
