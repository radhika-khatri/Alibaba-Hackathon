import os
import json
import requests


QWEN_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
QWEN_CHAT_MODEL = os.getenv("QWEN_CHAT_MODEL", "qwen-plus")
QWEN_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen-vl-plus")
QWEN_EMBED_MODEL = os.getenv("QWEN_EMBED_MODEL", "text-embedding-v1")


HEADERS = {
"Authorization": f"Bearer {QWEN_API_KEY}",
"Content-Type": "application/json"
}




def qwen_vl_extract(image_url: str) -> dict:
    messages = [
        {"role": "system", "content": "Extract fields from images and ALWAYS return strict JSON only."},
        {
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": {"url": image_url}},
                {"type": "text", "text": (
                "Extract keys: order_id, tracking_no, product_name, price, issue_type, confidence. "
                "Return strict JSON only. Use null for missing. If confidence < 0.7 add low_confidence: true."
                )}
            ]
        }
    ]
    payload = {"model": QWEN_VL_MODEL, "messages": messages, "temperature": 0.1}
    r = requests.post(f"{QWEN_BASE_URL}/chat/completions", headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"].strip()
    try:
        if "{" in content and "}" in content:
            content = content[content.find("{"): content.rfind("}")+1]
        data = json.loads(content)
    except Exception:
        data = {}
    return data




def qwen_embed(texts: list[str]) -> list[list[float]]:
    payload = {"model": QWEN_EMBED_MODEL, "input": texts}
    r = requests.post(f"{QWEN_BASE_URL}/embeddings", headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return [item["embedding"] for item in data.get("data", [])]




def qwen_chat(messages: list[dict]) -> str:
    payload = {"model": QWEN_CHAT_MODEL, "messages": messages, "temperature": 0.2}
    r = requests.post(f"{QWEN_BASE_URL}/chat/completions", headers=HEADERS, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]