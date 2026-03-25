"""
ASGI entry for Render / local runs: `uvicorn main:app --host 0.0.0.0 --port $PORT`
The FastAPI application lives in prediction.py.
"""

from prediction import app

__all__ = ["app"]
