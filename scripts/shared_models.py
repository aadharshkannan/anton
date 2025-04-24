from pydantic import BaseModel, Field
from typing import List, Iterable

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class Exchange(BaseModel):
    time: str = Field(..., description="Timestamp in original WhatsApp format")
    author: str
    message: str

class Session(BaseModel):
    session_start: str
    session_end: str
    exchanges: List[Exchange]

class TrainingSnippet(BaseModel):
    context: str
    output: str    