"""decide.py: the threshold rule on small made-up scores, and the recorded decisions on the stored development results."""
import json
import math

import pytest

import decide

RECORDED = json.loads(decide.OUT.read_text())


def scores(*pairs):
    """{image: (class, probability of 'AI')}, as decide.ai_scores returns them."""
    return dict(enumerate(pairs))


def test_a_score_at_the_threshold_counts_as_ai():
    assert decide.accuracy(scores(("ai", 0.5), ("real", 0.49)), 0.5) == 100


def test_best_threshold_lies_halfway_between_scores():
    # 0.5 gets 2 of 3 right; halfway between 0.7 and 0.9 all 3 are right
    assert decide.best_threshold(scores(("ai", 0.9), ("real", 0.7), ("real", 0.2))) == 0.8


def test_best_threshold_stays_at_0_5_when_nothing_is_better():
    # 0.35 is as good as 0.5, so the default is kept
    assert decide.best_threshold(scores(("ai", 0.9), ("ai", 0.6), ("real", 0.1))) == 0.5


def test_best_threshold_takes_the_candidate_closest_to_0_5_among_equals():
    # 0.325 and 0.7 both get 3 of 4 right; 0.325 is closer to 0.5
    assert decide.best_threshold(scores(("real", 0.2), ("ai", 0.45), ("real", 0.6), ("ai", 0.8))) == pytest.approx(0.325)


def test_recorded_decisions_are_reproduced_from_the_database():
    decide.check()  # the three rules again on results/results.db; fails if any figure of results/decisions.json differs


@pytest.mark.parametrize("model", sorted(RECORDED["threshold"]))
def test_shift_of_the_real_logit_moves_the_models_own_answer_to_the_threshold(model):
    chosen = RECORDED["threshold"][model]
    t, shift = chosen["threshold_on_ai_score"], chosen["shift_of_real_logit"]
    assert chosen["threshold_on_real_score"] == pytest.approx(1 - t)
    for p in (t - 0.001, t + 0.001):  # probability of "AI" just below and just above the threshold
        ai_logit, real_logit = math.log(p), math.log(1 - p)  # logits whose softmax gives p
        assert (ai_logit > real_logit + shift) == (p > t)
