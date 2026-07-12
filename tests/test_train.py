from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from summar.train import _remove_intermediate_checkpoints, build_parser


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingArgumentsTests(unittest.TestCase):
    def test_improved_training_defaults(self) -> None:
        args = build_parser().parse_args([])
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertEqual(args.warmup_ratio, 0.1)
        self.assertEqual(args.label_smoothing_factor, 0.1)
        self.assertEqual(args.early_stopping_patience, 6)
        self.assertEqual(args.eval_steps, 40)
        self.assertEqual(args.save_steps, 40)

    def test_learning_rate_can_be_overridden(self) -> None:
        args = build_parser().parse_args(["--learning-rate", "0.0001"])
        self.assertEqual(args.learning_rate, 1e-4)

    def test_intermediate_checkpoints_are_removed(self) -> None:
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as directory:
            output_dir = Path(directory)
            (output_dir / "checkpoint-40").mkdir()
            (output_dir / "checkpoint-final").mkdir()
            (output_dir / "model.safetensors").touch()

            removed = _remove_intermediate_checkpoints(output_dir)

            self.assertEqual(removed, 1)
            self.assertFalse((output_dir / "checkpoint-40").exists())
            self.assertTrue((output_dir / "checkpoint-final").exists())
            self.assertTrue((output_dir / "model.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
