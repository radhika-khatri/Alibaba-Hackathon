from typing import Optional, List
from pydantic import BaseModel, Field


class PresignRequest(BaseModel):
    content_type: str = Field(default="image/png")
    prefix: str = Field(default="uploads/")


class PresignResponse(BaseModel):
    upload_url: str
    object_key: str
    expires_in: int


class SubmitScreenshotRequest(BaseModel):
    user_id: str
    image_url: str
    initial_message: Optional[str] = None
    top_k: int = 4


class ExtractedFields(BaseModel):
    order_id: Optional[str] = None
    tracking_no: Optional[str] = None
    product_name: Optional[str] = None
    price: Optional[str] = None
    issue_type: Optional[str] = None
    confidence: Optional[float] = None
    low_confidence: Optional[bool] = None


class KBDoc(BaseModel):
    doc_id: str
    title: Optional[str] = None
    url: Optional[str] = None
    snippet: str
    score: float


class SubmitScreenshotResponse(BaseModel):
    extracted: ExtractedFields
    kb: List[KBDoc]
    answer: str


# For KB ingestion
class KBItem(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    url: Optional[str] = None
    text: str