"""
schemas.py — Modelos Pydantic para la API de corrección OCR.
"""

from pydantic import BaseModel, Field


class CorrectionRequest(BaseModel):
    """Solicitud de corrección de texto OCR."""

    text: str = Field(
        ...,
        description="Texto OCR ruidoso a corregir",
        min_length=1,
        max_length=10000,
        examples=["Ths artcle presens a novl approch to electrcal enginring"],
    )


class CorrectionResponse(BaseModel):
    """Respuesta con el texto corregido."""

    original: str = Field(description="Texto original enviado")
    corrected: str = Field(description="Texto corregido por el modelo")
