from __future__ import annotations

import argparse
import gc
import os
import shutil
from pathlib import Path

from summar.config import DEFAULT_MODEL_NAME, DEFAULT_OUTPUT_DIR, TrainingConfig


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_CACHE_DIR = PROJECT_ROOT / ".cache"
HF_HOME_DIR = LOCAL_CACHE_DIR / "hf"
HF_DATASETS_CACHE_DIR = LOCAL_CACHE_DIR / "datasets"
HF_METRICS_CACHE_DIR = LOCAL_CACHE_DIR / "metrics"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Обучение модели суммаризации")
    parser.add_argument("--dataset-name", type=str, help="Имя датасета из Hugging Face")
    parser.add_argument("--dataset-config", type=str, help="Конфиг датасета")
    parser.add_argument("--dataset-path", type=Path, help="Путь до одного локального json/csv файла")
    parser.add_argument("--train-file", type=Path, help="Путь до train json/csv файла")
    parser.add_argument("--validation-file", type=Path, help="Путь до validation json/csv файла")
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--validation-split", type=str, default="validation")
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--summary-column", type=str, default="summary")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--label-smoothing-factor", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--eval-steps", type=int, default=40)
    parser.add_argument("--save-steps", type=int, default=40)
    parser.add_argument(
        "--keep-intermediate-checkpoints",
        action="store_true",
        help="Не удалять checkpoint-* после сохранения итоговой модели",
    )
    parser.add_argument("--test-dir", type=Path, default=PROJECT_ROOT / "test_texts")
    parser.add_argument(
        "--skip-test-run",
        action="store_true",
        help="Не запускать постоянные тестовые тексты после обучения",
    )
    return parser


def _detect_loader(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    raise ValueError("Поддерживаются только .json и .csv")


def _load_dataset(args: argparse.Namespace):
    from datasets import load_dataset

    HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if args.train_file:
        loader = _detect_loader(args.train_file)
        data_files = {"train": str(args.train_file)}
        if args.validation_file:
            data_files["validation"] = str(args.validation_file)
        return load_dataset(loader, data_files=data_files, cache_dir=str(HF_DATASETS_CACHE_DIR))

    if args.dataset_path:
        loader = _detect_loader(args.dataset_path)
        return load_dataset(
            loader,
            data_files={"train": str(args.dataset_path)},
            cache_dir=str(HF_DATASETS_CACHE_DIR),
        )

    if args.dataset_name:
        if args.dataset_config:
            return load_dataset(
                args.dataset_name,
                args.dataset_config,
                cache_dir=str(HF_DATASETS_CACHE_DIR),
            )
        return load_dataset(args.dataset_name, cache_dir=str(HF_DATASETS_CACHE_DIR))

    raise ValueError(
        "Нужно указать --train-file, --dataset-path или --dataset-name"
    )


def _remove_intermediate_checkpoints(output_dir: Path) -> int:
    removed = 0
    for path in output_dir.glob("checkpoint-*"):
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit():
            shutil.rmtree(path)
            removed += 1
    return removed


def train(args: argparse.Namespace) -> None:
    HF_HOME_DIR.mkdir(parents=True, exist_ok=True)
    HF_DATASETS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    HF_METRICS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(HF_HOME_DIR)
    os.environ["HF_DATASETS_CACHE"] = str(HF_DATASETS_CACHE_DIR)
    os.environ["HF_EVALUATE_CACHE"] = str(HF_METRICS_CACHE_DIR)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    try:
        import numpy as np
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Для обучения нужно установить зависимости из requirements.txt"
        ) from exc

    config = TrainingConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing_factor,
        early_stopping_patience=args.early_stopping_patience,
        evaluation_steps=args.eval_steps,
        checkpoint_steps=args.save_steps,
    )

    dataset = _load_dataset(args)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    rouge = None
    try:
        import evaluate

        rouge = evaluate.load("rouge", cache_dir=str(HF_METRICS_CACHE_DIR))
    except Exception:
        rouge = None

    train_split = args.train_split if args.train_split in dataset else "train"
    validation_split = args.validation_split if args.validation_split in dataset else None
    source_columns = dataset[train_split].column_names

    def preprocess(batch: dict) -> dict:
        inputs = tokenizer(
            batch[args.text_column],
            max_length=config.max_input_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch[args.summary_column],
            max_length=config.max_target_length,
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=source_columns,
    )

    def compute_metrics(eval_pred):
        if rouge is None:
            return {}
        predictions, labels = eval_pred
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = rouge.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
            use_stemmer=True,
        )
        return {key: round(value, 4) for key, value in scores.items()}

    has_validation = validation_split is not None
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        label_smoothing_factor=config.label_smoothing_factor,
        save_total_limit=2,
        num_train_epochs=config.epochs,
        predict_with_generate=True,
        logging_steps=5,
        eval_strategy="steps" if has_validation else "no",
        eval_steps=config.evaluation_steps if has_validation else None,
        save_steps=config.checkpoint_steps,
        save_strategy="steps",
        load_best_model_at_end=has_validation,
        metric_for_best_model="eval_loss" if has_validation else None,
        greater_is_better=False if has_validation else None,
        fp16=False,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized[train_split],
        eval_dataset=tokenized[validation_split] if has_validation else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics if has_validation and rouge is not None else None,
        callbacks=(
            [EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)]
            if has_validation
            else None
        ),
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    if not args.keep_intermediate_checkpoints:
        removed = _remove_intermediate_checkpoints(Path(config.output_dir))
        if removed:
            print(f"Удалено промежуточных чекпоинтов: {removed}")

    if not args.skip_test_run:
        # Reload the saved checkpoint through the same pipeline used by main.py.
        del trainer
        del model
        gc.collect()

        from summar.evaluation import run_test_suite

        output_dir = Path(config.output_dir)
        test_output = output_dir / "test_results.json"
        print("Запуск постоянного тестового набора...")
        run_test_suite(
            model_name=str(output_dir),
            test_dir=args.test_dir,
            output_path=test_output,
        )
        print(f"Результаты тестов сохранены: {test_output}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
