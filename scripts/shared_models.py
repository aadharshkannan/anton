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

class HellaSwagEntry(BaseModel):
    """Pydantic representation of one HellaSwag example (5 endings)."""

    context: str
    ending0: str
    ending1: str
    ending2: str
    ending3: str
    ending4: str
    label: int

    @classmethod
    def from_endings(cls, context: str, endings: List[str], label: int) -> "HellaSwagEntry":
        assert len(endings) == 5, "Need exactly 5 endings (4 alt + original)."
        return cls(
            context=context,
            ending0=endings[0],
            ending1=endings[1],
            ending2=endings[2],
            ending3=endings[3],
            ending4=endings[4],
            label=label,
        )