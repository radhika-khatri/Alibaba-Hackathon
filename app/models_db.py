from sqlalchemy import Column, String, Text, JSON
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB
from datetime import datetime
from app.db import Base


class KBItemDB(Base):
    __tablename__ = "kb_items"
    id = Column(String, primary_key=True)
    title = Column(String, nullable=True)
    url = Column(String, nullable=True)
    text = Column(Text, nullable=False)


class TicketDB(Base):
    __tablename__ = "tickets"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    image_url = Column(String, nullable=False)
    extracted = Column(JSONB, nullable=False)
    answer = Column(Text, nullable=False)
    kb_doc_ids = Column(JSONB, nullable=False, server_default="[]")


class ChatMessageDB(Base):
    __tablename__ = "chat_messages"
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    role = Column(String, nullable=False)  # "user" or "bot"
    content_type = Column(String, nullable=False, default="text")  # "text", "video", "image"
    content = Column(Text, nullable=False)  # text or media URL
    meta_info = Column(JSON, nullable=True)  # optional extra info
    created_at = Column(TIMESTAMP, default=datetime.utcnow)