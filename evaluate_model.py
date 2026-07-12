from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from summar.config import DEFAULT_MODEL_NAME
from summar.evaluation import run_test_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка модели на постоянных тестовых текстах")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--test-dir", type=Path, default=Path("test_texts"))
    parser.add_argument("--output", type=Path, default=Path("results/test_results.json"))
    parser.add_argument("--device", default="cpu", help="cpu или cuda")
    args = parser.parse_args()
    run_test_suite(args.model_name, args.test_dir, args.output, args.device)
    print(f"Результаты сохранены: {args.output}")


if __name__ == "__main__":
    main()
