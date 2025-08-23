# =============================
# file: app/__init__.py
# =============================
# empty package marker


# =============================
# file: run.py (local dev entry)
# =============================
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)