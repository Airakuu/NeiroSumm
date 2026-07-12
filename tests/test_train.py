from __future__ import annotations

import unittest

from summar.train import build_parser


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


if __name__ == "__main__":
    unittest.main()
