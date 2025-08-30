# app/main.py
import os
import uuid
import json
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    PresignRequest, PresignResponse,
    SubmitScreenshotRequest, SubmitScreenshotResponse,
    ExtractedFields, KBDoc, KBItem
)
from app.oss_service import PresignParams, presign_upload
from app.qwen_service import qwen_vl_extract, qwen_embed, qwen_chat
from app.db import engine, Base, get_session
from app.models_db import KBItemDB, TicketDB

# ⬇️ NEW: use OpenSearch service instead of Pinecone
from app.opensearch_service import upsert_vectors, query_vector

load_dotenv()

app = FastAPI(title="Alibaba Support Agent API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    # Create tables if not exist (ApsaraDB RDS for PostgreSQL)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
        raise HTTPException(500, f"OSS presign failed: {e}")

# ---------- KB ingest (text -> embeddings -> OpenSearch + Postgres metadata) ----------
@app.post("/kb/upsert")
async def kb_upsert(items: List[KBItem], session: AsyncSession = Depends(get_session)):
    texts = [it.text for it in items]
    embs = qwen_embed(texts)  # not strictly needed by opensearch_service, but OK to keep

    vectors = []
    for it, emb in zip(items, embs):
        doc_id = it.id or uuid.uuid4().hex
        vectors.append({
            "id": doc_id,
            "values": emb,
            "metadata": {
                "title": it.title,
                "url": it.url,
                "text": it.text
            }
        })
        # upsert metadata in Postgres
        db_item = KBItemDB(id=doc_id, title=it.title, url=it.url, text=it.text)
        await session.merge(db_item)

    await session.commit()
    # store to Alibaba OpenSearch
    upsert_vectors(vectors)
    return {"ok": True, "count": len(vectors)}

# ---------- submit screenshot (Qwen-VL → embed → OpenSearch RAG → Qwen chat → save ticket) ----------
@app.post("/submit-screenshot", response_model=SubmitScreenshotResponse)
async def submit_screenshot(body: SubmitScreenshotRequest, session: AsyncSession = Depends(get_session)):
    # 1) Extract with Qwen-VL
    extracted = qwen_vl_extract(body.image_url)

    # 2) Build retrieval query
    parts = [body.initial_message or ""]
    for key in ("order_id", "tracking_no", "product_name", "issue_type"):
        val = extracted.get(key)
        if val:
            parts.append(f"{key}: {val}")
    query_text = "\n".join([p for p in parts if p]) or "customer support"

    # 3) Embedding + vector search (Alibaba OpenSearch)
    vec = qwen_embed([query_text])[0]
    search = query_vector(vec, top_k=body.top_k)

    kb_docs: List[KBDoc] = []
    kb_doc_ids: List[str] = []
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

    # 4) Final answer via Qwen chat (RAG)
    evidence_text = "\n\n".join([f"[Doc {i+1}] {d.title or ''}\n{d.snippet}" for i, d in enumerate(kb_docs)]) or "(no docs)"
    messages = [
        {"role": "system", "content": (
            "You are the official support assistant. Be concise and helpful. "
            "Use evidence when relevant. If low_confidence is true, ask a clarifying question first."
        )},
        {"role": "user", "content": (
            f"USER_MESSAGE:\n{body.initial_message or '(none)'}\n\n"
            f"EXTRACTED_FIELDS:\n{json.dumps(extracted, ensure_ascii=False)}\n\n"
            f"EVIDENCE:\n{evidence_text}"
        )}
    ]
    answer = qwen_chat(messages)

    # 5) Persist ticket in Postgres (ApsaraDB RDS)
    ticket = TicketDB(
        id=uuid.uuid4().hex,
        user_id=body.user_id,
        image_url=body.image_url,
        extracted=extracted,
        answer=answer,
        kb_doc_ids=kb_doc_ids,
    )
    session.add(ticket)
    await session.commit()

    ef = ExtractedFields(**{k: extracted.get(k) for k in ExtractedFields.model_fields})
    return SubmitScreenshotResponse(extracted=ef, kb=kb_docs, answer=answer)
