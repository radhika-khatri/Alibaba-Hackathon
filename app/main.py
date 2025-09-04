# app/main.py
import os
from dotenv import load_dotenv

load_dotenv()

import uuid
import logging
from typing import List
from fastapi import FastAPI, Depends, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession
import pandas as pd

from app.models import (
    PresignRequest, PresignResponse,
    SubmitScreenshotRequest, SubmitScreenshotResponse,
    ExtractedFields, KBDoc
)
from app.oss_service import PresignParams, presign_upload
from app.gemini_service import (
    gemini_text_extract,
    gemini_vl_extract,
    embed_texts,
    gemini_chat
)
from app.db import engine, Base, get_session
from app.models_db import KBItemDB, TicketDB, ChatMessageDB
from app.opensearch_service import upsert_vectors, query_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Alibaba Support Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.options("/{rest_of_path:path}")
async def preflight_handler(rest_of_path: str):
    return {"status": "OK"}


# ---------- health ----------
@app.get("/health")
async def health():
    return {"status": "ok"}


# ---------- presign upload ----------
@app.post("/presign-upload", response_model=PresignResponse)
async def api_presign(body: PresignRequest):
    try:
        result = presign_upload(PresignParams(content_type=body.content_type, prefix=body.prefix))
        return PresignResponse(**result)
    except Exception as e:
        logger.exception("OSS presign failed")
        raise HTTPException(status_code=500, detail=f"OSS presign failed: {e}")


# ---------- KB ingest ----------
@app.post("/kb/upload-files")
async def upload_kb_files(files: List[UploadFile] = File(...), session: AsyncSession = Depends(get_session)):
    items = []

    for file in files:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(".json"):
            df = pd.DataFrame(pd.read_json(file.file))
        else:
            continue

        for _, row in df.iterrows():
            text = row.get("text") or ""
            if not text.strip():
                continue
            items.append({
                "id": row.get("id") or str(uuid.uuid4().hex),
                "title": row.get("title"),
                "url": row.get("url"),
                "text": text
            })

    if not items:
        return {"ok": False, "count": 0, "message": "No valid text found"}

    # Generate embeddings
    texts = [it["text"] for it in items]
    embs = embed_texts(texts)
    logger.info(f"[KB Upsert] Generated embeddings for {len(embs)} items")

    vectors = []
    for it, emb in zip(items, embs):
        vectors.append({
            "id": it["id"],
            "values": emb,
            "metadata": {"title": it.get("title"), "url": it.get("url"), "text": it["text"]}
        })
        db_item = KBItemDB(id=it["id"], title=it.get("title"), url=it.get("url"), text=it["text"])
        await session.merge(db_item)

    await session.commit()
    logger.info(f"[KB Upsert] Committed {len(items)} items to Postgres")

    try:
        upsert_vectors(vectors)
        logger.info(f"[KB Upsert] Upserted {len(vectors)} vectors to OpenSearch")
    except Exception as e:
        logger.warning(f"[KB Upsert] OpenSearch upsert failed: {e}")

    return {"ok": True, "count": len(items)}


# ---------- submit screenshot ----------
@app.post("/submit-screenshot", response_model=SubmitScreenshotResponse)
async def submit_screenshot(request: Request, session: AsyncSession = Depends(get_session)):
    try:
        body_data = await request.json()
        body = SubmitScreenshotRequest(**body_data)
        logger.info(f"Received request: {body.dict()}")

        # Extract fields
        extracted = {}
        try:
            if body.image_url:
                logger.info("[API] Using Gemini-VL for image extraction")
                extracted = gemini_vl_extract(body.image_url)
            else:
                logger.info("[API] Using Gemini-Text for plain text extraction")
                extracted = gemini_text_extract(body.initial_message or "")
        except Exception as e:
            logger.warning(f"Field extraction failed: {e}")
            extracted = {}

        # Build retrieval query
        parts = [body.initial_message or ""]
        for key in ("order_id", "tracking_no", "product_name", "issue_type"):
            val = extracted.get(key)
            if val:
                parts.append(f"{key}: {val}")
        query_text = "\n".join([p for p in parts if p]) or "customer support"

        # Generate embedding
        vec = embed_texts([query_text])[0]

        kb_docs, kb_doc_ids = [], []

        try:
            search = query_vector(vec, top_k=body.top_k)
            for m in search.matches or []:
                meta = m.metadata or {}
                kb_docs.append(KBDoc(
                    doc_id=m.id,
                    title=meta.get("title"),
                    url=meta.get("url"),
                    snippet=(meta.get("text") or "")[:500],
                    score=float(m.score or 0.0)
                ))
                kb_doc_ids.append(m.id)
        except Exception as e:
            logger.warning(f"[API] OpenSearch failed: {e}")
            kb_docs, kb_doc_ids = [], []

        # Prepare messages for chatbot
        evidence_text = "\n\n".join([f"[Doc {i+1}] {d.title or ''}\n{d.snippet}" for i, d in enumerate(kb_docs)])
        messages = [
            {"role": "system", "content": (
                "You are a friendly support assistant. Respond naturally to the user, "
                "even if no knowledge base info is available."
            )},
            {"role": "user", "content": f"{body.initial_message or ''}\nEVIDENCE:\n{evidence_text}"}
        ]

        try:
            answer_text = gemini_chat(messages)
            if not answer_text or "something went wrong" in answer_text.lower():
                raise Exception("Invalid response")
        except Exception:
            logger.warning("[API] Gemini chat failed, using fallback response")
            answer_text = f"{body.initial_message or 'Hello!'} Hello! How are you today?"

        # Persist ticket
        ticket = TicketDB(
            id=uuid.uuid4().hex,
            user_id=body.user_id,
            image_url=body.image_url,
            extracted=extracted,
            answer=answer_text,
            kb_doc_ids=kb_doc_ids,
        )
        session.add(ticket)
        await session.commit()

        # Save chat message history
        chat_message = ChatMessageDB(
            id=uuid.uuid4().hex,
            user_id=body.user_id,
            role="bot",
            content_type="text",
            content=answer_text
        )
        session.add(chat_message)
        await session.commit()

        ef = ExtractedFields(**{k: extracted.get(k) for k in ExtractedFields.model_fields})
        return SubmitScreenshotResponse(
            extracted=ef,
            kb=kb_docs,
            answer=answer_text
        )

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return SubmitScreenshotResponse(
            extracted=ExtractedFields(),
            kb=[],
            answer="Hello! How can I help you today?"
        )
