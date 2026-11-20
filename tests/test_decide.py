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


# Third round (PROTOCOL.md, decision 2): a candidate qualifies with new >= 73.40, old >= 81.35, old AI >= 72.97 and
# JPEG >= 74.67; among those the highest mean of old and new; ties: higher JPEG, then A before B before C.
GOOD = {"new": 75.0, "old": 82.0, "old_ai": 74.0, "jpeg": 75.0}


def candidate(**changes):
    f = {**GOOD, **changes}
    return {**f, "mean_old_new": (f["old"] + f["new"]) / 2}


def test_the_limits_are_those_of_the_protocol():
    assert decide.LIMITS3 == {"new": 73.40, "old": 81.35, "old_ai": 72.97, "jpeg": 74.67}


def test_the_baseline_of_the_protocol_is_reproduced_from_the_database():
    record, _ = decide.figures3("cnn_s5_raw")
    assert round(record["threshold_on_ai_score"], 4) == 0.6594
    assert {k: record[k] for k in ("old", "old_ai", "new", "jpeg")} == {"old": 83.35, "old_ai": 77.97, "new": 68.40, "jpeg": 76.67}


@pytest.mark.parametrize("figure, at_limit, below", [
    ("new", 73.40, 73.39), ("old", 81.35, 81.34), ("old_ai", 72.97, 72.96), ("jpeg", 74.67, 74.66)])
def test_each_limit_is_inclusive_and_each_is_needed(figure, at_limit, below):
    assert decide.qualifies(candidate(**{figure: at_limit}))
    assert not decide.qualifies(candidate(**{figure: below}))


def test_the_current_cnn_itself_would_not_qualify():
    assert not decide.qualifies(candidate(new=68.40, old=83.35, old_ai=77.97, jpeg=76.67))


def test_the_highest_mean_of_old_and_new_is_adopted():
    assert decide.choose({"A": candidate(new=74.0), "B": candidate(new=80.0), "C": candidate(new=76.0)}) == "B"


def test_a_candidate_that_forgets_the_old_kinds_is_not_adopted_however_good_on_the_new():
    assert decide.choose({"A": candidate(new=74.0), "B": candidate(new=95.0, old_ai=70.0), "C": candidate(new=95.0, old=80.0)}) == "A"


def test_ties_go_to_the_higher_jpeg_figure_then_to_the_order_a_b_c():
    assert decide.choose({"A": candidate(jpeg=75.0), "B": candidate(jpeg=76.0), "C": candidate(jpeg=76.0)}) == "B"
    assert decide.choose({"A": candidate(), "B": candidate(), "C": candidate()}) == "A"


def test_no_candidate_qualifies_and_none_is_adopted():
    assert decide.choose({"A": candidate(new=70.0), "B": candidate(jpeg=70.0), "C": candidate(old=80.0)}) is None


def test_fusion_weight_by_the_mean_of_old_and_new_then_jpeg_then_the_smaller_weight():
    tried = {0.25: {"mean_old_new": 80.0, "jpeg": 75.0}, 0.5: {"mean_old_new": 82.0, "jpeg": 70.0}, 0.75: {"mean_old_new": 82.0, "jpeg": 71.0}}
    assert decide.pick_weight(tried) == 0.75
    assert decide.pick_weight({0.25: {"mean_old_new": 82.0, "jpeg": 71.0}, 0.5: {"mean_old_new": 82.0, "jpeg": 71.0}}) == 0.25
