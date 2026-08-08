"""
schemas.py — Pydantic models for the OCR correction API.
"""

from pydantic import BaseModel, Field


class CorrectionRequest(BaseModel):
    """Request to correct a noisy OCR text."""

    text: str = Field(
        ...,
        description="Noisy OCR text to correct",
        min_length=1,
        max_length=10000,
        examples=["Ths artcle presens a novl approch to electrcal enginring"],
    )


class CorrectionResponse(BaseModel):
    """Response containing the corrected text."""

    original: str = Field(description="Original text that was submitted")
    corrected: str = Field(description="Text corrected by the model")
