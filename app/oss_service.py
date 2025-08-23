import os
import uuid
import oss2
from pydantic import BaseModel


OSS_ENDPOINT = os.getenv("OSS_ENDPOINT")
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET")
OSS_BUCKET = os.getenv("OSS_BUCKET")


class PresignParams(BaseModel):
    content_type: str
    prefix: str = "uploads/"


def presign_upload(params: PresignParams):
    auth = oss2.Auth(OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET)
    bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)
    object_key = f"{params.prefix}{uuid.uuid4().hex}"
    headers = {"Content-Type": params.content_type}
    url = bucket.sign_url("PUT", object_key, 300, headers=headers)
    return {
    "upload_url": url,
    "object_key": object_key,
    "expires_in": 300
    }