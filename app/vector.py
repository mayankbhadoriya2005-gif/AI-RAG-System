
import logging
from functools import lru_cache
from pathlib import Path

import chromadb

from app.config import settings
from app.observability import trace

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def client():
    s = settings()

    chroma_path = Path(s.chroma_path).resolve()
    chroma_path.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=str(chroma_path))


def collection(tenant):
    name = "tenant_" + tenant.replace("-", "_")

    return client().get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )


def namespace(tenant):
    return "tenant-" + tenant


@trace("upsert")
def upsert(tenant, records):
    col = collection(tenant)

    ids = [r["id"] for r in records]
    embeddings = [r["values"] for r in records]
    metadatas = [r["metadata"] for r in records]

    col.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )


@trace("search")
def search(tenant, vector, groups, owner, limit=5):
    col = collection(tenant)

    result = col.query(
        query_embeddings=[vector],
        n_results=limit,
        where={
            "$or": [
                {"allowed_groups": {"$in": list(groups)}},
                {"owner_sub": owner}
            ]
        }
    )

    matches = []

    for i in range(len(result["ids"][0])):
        matches.append({
            "id": result["ids"][0][i],
            "score": 1 - result["distances"][0][i],
            "metadata": result["metadatas"][0][i] or {}
        })

    return matches


def delete_document(tenant, ids):
    if ids:
        collection(tenant).delete(ids=ids)
