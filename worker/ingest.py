
import io
import logging
from pathlib import Path

from docx import Document
from pypdf import PdfReader

from app.aws import embed
from app.vector import upsert
from app.config import settings
from app.db import connection, audit
from app.observability import trace, initialize

initialize()
log = logging.getLogger(__name__)


def upload_directory():
    path = Path(settings().upload_dir).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse(blob: bytes, suffix: str):
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(blob))
        return [
            (i + 1, page.extract_text() or "")
            for i, page in enumerate(reader.pages)
        ]

    if suffix == ".docx":
        doc = Document(io.BytesIO(blob))
        return [(1, "\n".join(p.text for p in doc.paragraphs))]

    return [(1, blob.decode("utf-8"))]


def chunks(text, size=1200, overlap=180):
    text = " ".join(text.split())
    pos = 0

    while pos < len(text):
        yield text[pos:pos + size]

        if pos + size >= len(text):
            break

        pos += size - overlap


@trace("process_local_file")
def process_s3_key(key):
    s = settings()

    base_dir = upload_directory()
    file_path = (base_dir / key).resolve()

    if not file_path.is_relative_to(base_dir):
        raise ValueError("Invalid upload file path")

    with connection() as conn:
        doc = conn.execute(
            "SELECT * FROM documents WHERE s3_key=%s FOR UPDATE",
            (key,)
        ).fetchone()

        if not doc or doc["status"] == "ready":
            return

        if doc["status"] not in ("pending", "failed"):
            return

        conn.execute(
            "UPDATE documents SET status='processing' WHERE id=%s",
            (doc["id"],)
        )

    try:
        if not file_path.is_file():
            raise FileNotFoundError("Uploaded file not found")

        if file_path.stat().st_size > s.max_upload_bytes:
            raise ValueError("File exceeds maximum upload size")

        blob = file_path.read_bytes()

        if len(blob) > s.max_upload_bytes:
            raise ValueError("File exceeds maximum upload size")

        suffix = file_path.suffix.lower()

        if suffix == ".pdf" and not blob.startswith(b"%PDF"):
            raise ValueError("Invalid PDF file")

        pages = parse(blob, suffix)

        parts = [
            (page, part)
            for page, page_text in pages
            for part in chunks(page_text)
        ]

        if not parts:
            raise ValueError("No readable text found in document")

        vectors = []

        for i, (page, part) in enumerate(parts):
            vectors.append({
                "id": f'{doc["id"]}:{doc["current_version"]}:{i}',
                "values": embed(part),
                "metadata": {
                    "document_id": str(doc["id"]),
                    "title": doc["title"],
                    "page": page,
                    "text": part,
                    "owner_sub": doc["owner_sub"],
                    "allowed_groups": doc["allowed_groups"]
                }
            })

        for i in range(0, len(vectors), 50):
            upsert(
                doc["tenant_id"],
                vectors[i:i + 50]
            )

        with connection() as conn:
            conn.execute(
                "UPDATE documents SET status='ready' WHERE id=%s",
                (doc["id"],)
            )
            audit(
                conn,
                doc["owner_sub"],
                doc["tenant_id"],
                "document.indexed",
                {"document_id": str(doc["id"])}
            )

        log.info("Document indexed successfully: %s", key)

    except Exception:
        log.exception("Document ingestion failed: %s", key)

        with connection() as conn:
            conn.execute(
                "UPDATE documents SET status='failed' WHERE id=%s",
                (doc["id"],)
            )

        raise
