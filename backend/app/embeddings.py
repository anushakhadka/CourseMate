import os
import httpx

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

async def embed(texts: list[str]) -> list[list[float]]:
    # No key? Return random vectors so the pipeline still works.
    if not OPENAI_API_KEY:
        import random
        return [[random.random() for _ in range(1536)] for _ in texts]

    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=headers, json={"model": MODEL, "input": texts})
        resp.raise_for_status()
        data = resp.json()
        return [d["embedding"] for d in data["data"]]
