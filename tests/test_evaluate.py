"""Focused tests for evaluation metrics and confidence intervals."""

from __future__ import annotations

import unittest

import numpy as np

from training.evaluate import _bootstrap


class BootstrapTests(unittest.TestCase):
    def test_confidence_intervals_exclude_categorical_metrics(self) -> None:
        scores = {
            "signer_a": {
                "gen": np.array([0.9]),
                "skilled": np.array([0.1]),
                "random": {},
            },
            "signer_b": {
                "gen": np.array([0.8]),
                "skilled": np.array([0.2]),
                "random": {},
            },
        }

        confidence = _bootstrap(scores, ["signer_a", "signer_b"], 0.5, 10)

        self.assertIsNotNone(confidence)
        assert confidence is not None
        self.assertNotIn("negatives_source", confidence)
        self.assertIn("accuracy", confidence)
