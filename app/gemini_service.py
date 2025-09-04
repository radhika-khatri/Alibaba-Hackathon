# app/gemini_service.py
import os
import requests
import logging
from typing import List, Dict

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ---- ENV VARS ----
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_ENDPOINT = os.getenv(
    "GEMINI_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
)
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash")
GEMINI_VL_MODEL = os.getenv("GEMINI_VL_MODEL", "gemini-2.0-flash-vl")

HEADERS = {
    "Content-Type": "application/json",
    "X-goog-api-key": GEMINI_API_KEY,  # Use X-goog-api-key instead of Authorization
}

# ---- Embedding model (SentenceTransformers) ----
MODEL_NAME = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
_embed_model = None  # Lazy-load

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[gemini] Loading SentenceTransformer model: {MODEL_NAME}")
        _embed_model = SentenceTransformer(MODEL_NAME)
    return _embed_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_embed_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    logger.debug(
        f"[embed_texts] Generated {len(embeddings)} embeddings, "
        f"dim={len(embeddings[0]) if len(embeddings) > 0 else 0}"
    )
    return embeddings.tolist()

# ---- Text-only extraction / completion ----
def gemini_text_extract(text: str) -> Dict:
    if not text.strip():
        return {}

    payload = {
        "contents": [
            {
                "parts": [{"text": text}]
            }
        ]
    }

    try:
        logger.info(f"[gemini_text_extract] Sending request to Gemini endpoint")
        r = requests.post(GEMINI_ENDPOINT, headers=HEADERS, json=payload, timeout=30)
        r.raise_for_status()
        try:
            return r.json()
        except ValueError:
            logger.error(f"[gemini_text_extract] Invalid JSON: {r.text}")
            return {"error": "Invalid response from Gemini", "raw_text": r.text}
    except Exception as e:
        logger.error(f"[gemini_text_extract] Failed: {e}")
        return {"error": str(e)}

# ---- Optional: image + text extraction ----
def gemini_vl_extract(image_url: str) -> Dict:
    """
    Extract information from an image using Gemini Vision.
    Downloads the image and converts to base64 since Gemini doesn't support direct URL access.
    """
    try:
        logger.info(f"[gemini_vl_extract] Downloading image from: {image_url}")
        
        # Download the image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Convert to base64
        import base64
        image_data = base64.b64encode(response.content).decode('utf-8')
        mime_type = "image/jpeg"  # You might want to detect this from response headers
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": "Extract order_id, tracking_no, product_name, price, issue_type from this e-commerce/shipping image. Return JSON with null for missing values. Only return the JSON object, no other text."
                        },
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": image_data
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 512,
                "responseMimeType": "application/json"
            }
        }
        
        logger.info("[gemini_vl_extract] Sending request to Gemini Vision API")
        r = requests.post(GEMINI_ENDPOINT, headers=HEADERS, json=payload, timeout=60)
        r.raise_for_status()
        
        result = r.json()
        logger.debug(f"[gemini_vl_extract] API response: {result}")
        return result
        
    except requests.exceptions.RequestException as e:
        logger.error(f"[gemini_vl_extract] Failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
            except:
                logger.error(f"Response text: {e.response.text}")
        return {}
    except Exception as e:
        logger.error(f"[gemini_vl_extract] Unexpected error: {e}")
        return {}

def gemini_chat(messages: list) -> str:
    """
    Send a chat request to Gemini-2.0-flash and return the model's response text.
    Gemini doesn't support 'system' role, so we need to convert system messages to user messages.
    """
    # Convert messages to Gemini format
    contents = []
    
    for msg in messages:
        if isinstance(msg, str):
            contents.append({
                "role": "user",
                "parts": [{"text": msg}]
            })
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Gemini doesn't support 'system' role, convert to 'user'
            if role == "system":
                role = "user"
            
            contents.append({
                "role": role,
                "parts": [{"text": content}]
            })
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.7,
            "maxOutputTokens": 512,
            "topP": 0.8,
            "topK": 40
        }
    }

    try:
        logger.info("[gemini_chat] Sending chat request to Gemini")
        r = requests.post(GEMINI_ENDPOINT, headers=HEADERS, json=payload, timeout=60)
        r.raise_for_status()
        resp = r.json()
        logger.debug(f"[gemini_chat] API response: {resp}")

        # Extract response text
        if "candidates" in resp and resp["candidates"]:
            candidate = resp["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                return " ".join([part.get("text", "") for part in candidate["content"]["parts"]])
        
        return "No content returned from Gemini."

    except requests.exceptions.RequestException as e:
        logger.error(f"[gemini_chat] Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                logger.error(f"Error details: {json.dumps(error_details, indent=2)}")
            except:
                logger.error(f"Response text: {e.response.text}")
        return "Failed to connect to Gemini service."






