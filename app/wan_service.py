import os
import requests

WAN_API_KEY = os.getenv("WAN_API_KEY")
WAN_BASE_URL = os.getenv("WAN_BASE_URL")

HEADERS = {"Authorization": f"Bearer {WAN_API_KEY}"}

def generate_video_from_text(text_prompt: str) -> str:
    payload = {"text": text_prompt, "resolution": "720p", "frame_rate": 30, "type": "video"}
    r = requests.post(f"{WAN_BASE_URL}/generate", headers=HEADERS, json=payload, timeout=120)
    r.raise_for_status()
    return r.json().get("video_url")

def generate_image_from_text(text_prompt: str) -> str:
    payload = {"text": text_prompt, "resolution": "1024x1024", "type": "image"}
    r = requests.post(f"{WAN_BASE_URL}/generate", headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    return r.json().get("image_url")
