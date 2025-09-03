# app/main.py
import os
from dotenv import load_dotenv

load_dotenv()

from pydantic import ValidationError
import uuid
import json
from typing import List

from fastapi import FastAPI, Depends, HTTPException, Request
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
from app.wan_service import generate_video_from_text, generate_image_from_text
from app.models_db import ChatMessageDB

from fastapi import UploadFile, File
import pandas as pd
from typing import List
import logging

logging.basicConfig(level=logging.DEBUG)
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
    # Create tables if not exist (ApsaraDB RDS for PostgreSQL)
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
        import traceback
        print(traceback.format_exc())  # <-- print full error
        raise HTTPException(status_code=500, detail=f"OSS presign failed: {e}")


# ---------- KB ingest (text -> embeddings -> OpenSearch + Postgres metadata) ----------
@app.post("/kb/upload-files")
async def upload_kb_files(
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(get_session)
):
    items = []

    for file in files:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(".json"):
            df = pd.DataFrame(pd.read_json(file.file))
        else:
            continue  # skip unsupported files

        for _, row in df.iterrows():
            text = row.get("text") or ""
            if not text.strip():
                continue  # skip rows without text
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
    embs = qwen_embed(texts)

    # Store in Postgres + OpenSearch
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
    upsert_vectors(vectors)

    return {"ok": True, "count": len(items)}
# ---------- submit screenshot (Qwen-VL → embed → OpenSearch RAG → Qwen chat → save ticket) ----------
@app.post("/submit-screenshot", response_model=SubmitScreenshotResponse)
async def submit_screenshot(request: Request, session: AsyncSession = Depends(get_session)):
    try:
        # Parse the JSON body
        body_data = await request.json()
        
        # Validate the request against our model
        body = SubmitScreenshotRequest(**body_data)
        
        print(f"Received request: {body.dict()}")
        
        # 1) Extract fields from image
        extracted = qwen_vl_extract(body.image_url)

        # 2) Build retrieval query and get evidence
        parts = [body.initial_message or ""]
        for key in ("order_id", "tracking_no", "product_name", "issue_type"):
            val = extracted.get(key)
            if val:
                parts.append(f"{key}: {val}")
        query_text = "\n".join([p for p in parts if p]) or "customer support"

        vec = qwen_embed([query_text])[0]
        search = query_vector(vec, top_k=body.top_k)

        kb_docs = []
        kb_doc_ids = []
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

        # 3) Generate Qwen chat answer
        evidence_text = "\n\n".join([f"[Doc {i+1}] {d.title or ''}\n{d.snippet}" for i, d in enumerate(kb_docs)]) or "(no docs)"
        messages = [
            {"role": "system", "content": (
                "You are the official support assistant. Be concise and helpful. "
                "Use evidence when relevant."
            )},
            {"role": "user", "content": (
                f"USER_MESSAGE:\n{body.initial_message or '(none)'}\n\n"
                f"EXTRACTED_FIELDS:\n{json.dumps(extracted, ensure_ascii=False)}\n\n"
                f"EVIDENCE:\n{evidence_text}"
            )}
        ]
        answer_text = qwen_chat(messages)

        # 4) Persist ticket
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

        # 5) Save chat message history (text)
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
        
    except ValidationError as e:
        print(f"Validation error: {e.errors()}")
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=str(e))