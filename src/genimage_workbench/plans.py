from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AdapterPlan:
    base_model: str
    dataset_name: str
    rank: int
    learning_rate: float
    steps: int
    precision: str
    validation_prompt: str
    seed: int


@dataclass(frozen=True)
class InferencePlan:
    prompt: str
    negative_prompt: str
    identity_id: str | None
    seed: int
    width: int
    height: int
    steps: int
    guidance_scale: float
    precision: str
    required_vram_gb: float
    scheduler: str


def create_adapter_plan(dataset_name: str, validation_prompt: str, seed: int = 42) -> AdapterPlan:
    if not dataset_name.strip() or not validation_prompt.strip():
        raise ValueError("dataset name and validation prompt are required")
    return AdapterPlan(
        base_model="stabilityai/stable-diffusion-xl-base-1.0",
        dataset_name=dataset_name,
        rank=16,
        learning_rate=1e-4,
        steps=1200,
        precision="bf16",
        validation_prompt=validation_prompt,
        seed=seed,
    )


def create_inference_plan(
    prompt: str,
    identity_id: str | None,
    available_vram_gb: float,
    seed: int = 42,
    requested_size: int = 1024,
) -> InferencePlan:
    if not prompt.strip():
        raise ValueError("prompt is required")
    if available_vram_gb < 6:
        raise ValueError("at least 6 GB of VRAM is required for local generation")

    max_size = 1024 if available_vram_gb >= 12 else 768
    size = min(requested_size, max_size)
    required_vram = 10.5 if size == 1024 else 7.5
    return InferencePlan(
        prompt=prompt,
        negative_prompt="low quality, distorted anatomy, duplicate character",
        identity_id=identity_id,
        seed=seed,
        width=size,
        height=size,
        steps=30,
        guidance_scale=6.5,
        precision="fp16",
        required_vram_gb=required_vram,
        scheduler="DPM++ 2M Karras",
    )


def plan_to_dict(plan: AdapterPlan | InferencePlan) -> dict[str, object]:
    return asdict(plan)
