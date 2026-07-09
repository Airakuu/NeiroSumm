from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from summar.config import DEFAULT_MODEL_NAME, SummarizerConfig
from summar.model import SummarizationModel


app = FastAPI(title="Summar API", version="0.1.0")


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model_name: str = Field(default=DEFAULT_MODEL_NAME)
    device: str = Field(default="cpu")


class SummarizeResponse(BaseModel):
    summary: str


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(payload: SummarizeRequest) -> SummarizeResponse:
    try:
        model = SummarizationModel(
            SummarizerConfig(model_name=payload.model_name, device=payload.device)
        )
        summary = model.summarize(payload.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return SummarizeResponse(summary=summary)
