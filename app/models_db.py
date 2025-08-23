from sqlalchemy import Column, String, Text
from sqlalchemy.dialects.postgresql import JSONB
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