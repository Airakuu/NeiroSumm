from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL_NAME = "checkpoints/rut5-absum-finetuned"
DEFAULT_OUTPUT_DIR = "checkpoints/summar-model"
DEFAULT_LOCAL_MODEL_DIR = Path("checkpoints") / "rut5-absum-finetuned"


@dataclass(slots=True)
class SummarizerConfig:
    model_name: str = DEFAULT_MODEL_NAME
    min_retention_ratio: float = 0.33
    max_input_tokens: int = 768
    max_summary_tokens: int = 256
    min_summary_tokens: int = 24
    num_beams: int = 4
    chunk_overlap: int = 1
    device: str = "cpu"


@dataclass(slots=True)
class TrainingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    output_dir: str = DEFAULT_OUTPUT_DIR
    max_input_length: int = 768
    max_target_length: int = 128
    learning_rate: float = 2e-5
    train_batch_size: int = 2
    eval_batch_size: int = 2
    epochs: int = 1
    weight_decay: float = 0.01
