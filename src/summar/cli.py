from __future__ import annotations

import argparse
from pathlib import Path

from summar.config import DEFAULT_MODEL_NAME, SummarizerConfig
from summar.model import SummarizationModel
from summar.preprocessing import normalize_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Суммаризация текста")
    parser.add_argument("--text", type=str, help="Текст для суммаризации")
    parser.add_argument("--input-file", type=Path, help="Путь до текстового файла")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Название модели Hugging Face",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu или cuda")
    return parser


def load_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.input_file:
        return args.input_file.read_text(encoding="utf-8")
    raise ValueError("Нужно передать --text или --input-file")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    source_text = load_text(args)
    config = SummarizerConfig(model_name=args.model_name, device=args.device)
    model = SummarizationModel(config)
    summary = model.summarize(source_text)

    source_length = len(normalize_text(source_text))
    summary_length = len(normalize_text(summary))
    reduction_percent = max(0.0, (1 - (summary_length / source_length)) * 100) if source_length else 0.0

    print(summary)
    print()
    print(f"Символов в исходном тексте: {source_length}")
    print(f"Символов в сокращенном тексте: {summary_length}")
    print(f"Процент сокращения: {reduction_percent:.1f}%")


if __name__ == "__main__":
    main()
