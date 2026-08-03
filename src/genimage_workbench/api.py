from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .dataset import DatasetRecord, record_to_dict, validate_manifest
from .plans import create_adapter_plan, create_inference_plan, plan_to_dict

app = FastAPI(title="Generative Image Workbench", version="0.1.0")


class DatasetRecordInput(BaseModel):
    image_path: str
    caption: str
    split: str
    identity_id: str | None = None
    reference_image_path: str | None = None


class AdapterPlanInput(BaseModel):
    dataset_name: str
    validation_prompt: str
    seed: int = 42


class InferencePlanInput(BaseModel):
    prompt: str
    identity_id: str | None = None
    available_vram_gb: float = Field(ge=1)
    seed: int = 42
    requested_size: int = Field(default=1024, ge=512, le=2048)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "generative-image-workbench"}


@app.post("/dataset/validate")
def validate_dataset(records: list[DatasetRecordInput]) -> dict[str, object]:
    domain_records = [DatasetRecord(**record.model_dump()) for record in records]
    report = validate_manifest(domain_records)
    return {"report": report.__dict__, "records": [record_to_dict(record) for record in domain_records]}


@app.post("/adapters/plan")
def adapter_plan(payload: AdapterPlanInput) -> dict[str, object]:
    try:
        return plan_to_dict(create_adapter_plan(**payload.model_dump()))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.post("/inference/plan")
def inference_plan(payload: InferencePlanInput) -> dict[str, object]:
    try:
        return plan_to_dict(create_inference_plan(**payload.model_dump()))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
