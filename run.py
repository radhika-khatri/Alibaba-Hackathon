# =============================
# file: app/__init__.py
# =============================
# empty package marker


# =============================
# file: run.py (local dev entry)
# =============================
import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="localhost", port=8000, reload=True)