"""fusion.py: the combined score is rescaled so that the threshold chosen on development data lies at 0.5."""
import json
from pathlib import Path

import pytest

from fusion import answer

THRESHOLD = json.loads(Path("results/decisions.json").read_text())["fusion"]["threshold_on_ai_score"]  # 0.250962
GRID = [i / 1000 for i in range(1001)]


@pytest.mark.parametrize("threshold", [THRESHOLD, 0.3, 0.5, 0.8])
def test_the_threshold_becomes_an_undecided_0_5_and_the_ends_stay(threshold):
    assert answer(threshold, threshold) == ("ai", 0.5)
    assert answer(0.0, threshold) == ("real", 1.0) and answer(1.0, threshold) == ("ai", 1.0)


def test_the_answer_is_ai_exactly_from_the_threshold_on_and_the_score_lies_between_0_5_and_1():
    for p in GRID:
        kind, score = answer(p, THRESHOLD)
        assert kind == ("ai" if p >= THRESHOLD else "real") and 0.5 <= score <= 1


def test_rescaling_keeps_the_order_of_the_images():
    ai = [score if kind == "ai" else 1 - score for kind, score in (answer(p, THRESHOLD) for p in GRID)]
    assert all(a < b for a, b in zip(ai, ai[1:]))
