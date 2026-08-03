# Generative Image Workbench

A production-minded foundation for preparing datasets, planning lightweight adapter fine-tuning, and enforcing repeatable local image-inference controls.

The project deliberately separates **data curation**, **training configuration**, and **inference policy**. This makes each decision inspectable before expensive GPU work begins and provides a clear starting point for integrating open-source diffusion pipelines.

## What it provides

- Dataset manifest validation for image, caption, identity, and split metadata
- Dataset quality summaries that flag missing identity labels and duplicate image paths
- LoRA-style adapter plans with explicit training, validation, and reproducibility settings
- Local-inference plans with VRAM budgets, deterministic seeds, resolution guardrails, and character-consistency controls
- Optional PyTorch/CUDA runtime probing for selecting a safe local-inference precision and GPU budget
- A small FastAPI surface for validating manifests and creating plans
- Docker, CI, and standard-library unit tests

## Architecture

```text
Dataset records -> manifest validation -> adapter plan -> inference plan -> local diffusion runner
                                      \-> quality report        \-> audit-ready metadata
```

The workbench does not bundle model weights or a training loop. Instead, it creates validated, serializable plans that a Diffusers, ComfyUI, or internal runner can execute. Keeping the model runner behind this boundary allows teams to swap models and hardware without changing dataset or product contracts.

## Quick start

```bash
python -m unittest discover -s tests -v
pip install -e .
uvicorn genimage_workbench.api:app --reload
```

To connect a local Diffusers runner, install the optional model stack with `pip install -e ".[ml]"`. Model weights and datasets are intentionally excluded from version control.

Example requests:

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/inference/plan \
  -H "Content-Type: application/json" \
  -d '{"prompt":"stylised explorer, full body", "identity_id":"explorer_01", "available_vram_gb":12}'
```

## Engineering choices

- **Character consistency:** identity labels and reference-image paths are first-class dataset fields, then carried into inference metadata.
- **Reproducibility:** plans require an explicit seed and capture scheduler, resolution, precision, and safety configuration.
- **Local inference safety:** requested resolution is constrained by available VRAM and high-risk requests are downgraded before a runner is invoked.
- **Data quality:** invalid records fail early so training and evaluation are not built on silently incomplete data.
- **Extensibility:** the API returns typed JSON plans and leaves framework-specific execution to an adapter.

## Scope and next steps

The next implementation layer would connect `AdapterPlan` to a PEFT/Diffusers LoRA trainer and `InferencePlan` to a GPU worker queue. That integration should include image quality evaluation, human review samples, metrics for identity consistency, and model-version tracking.
