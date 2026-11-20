"""Take the decisions of PROTOCOL.md from the results database and record them in results/decisions.json.

  python decide.py setting      # rule 1: does setting 5 replace setting 4?   (needs cnn_s5_raw on the development data)
  python decide.py threshold    # rule 2: decision threshold of a CNN          (needs cnn_s5_raw; --model for another run)
  python decide.py fusion       # rule 3: combination with Community Forensics (needs cnn_s5 with its threshold built in)
  python decide.py check        # are the recorded decisions reproduced from the database? (run by the GitHub workflow)

Third round (PROTOCOL.md, "Third round"): the five sets scored in the first two rounds are the development data.
  python decide.py round3         # decisions 1 and 2: thresholds of the current CNN and the candidates; which candidate
  python decide.py round3_fusion  # decision 4: combination of the adopted candidate (threshold built in) with Community Forensics

Rules 1 to 3 read only the check set and the development test set; the third-round rules read all five development sets.
Run `python results_db.py` first.
"""
import argparse
import json
import math
import sqlite3
from pathlib import Path

DEV = ("check", "main")
OUT = Path("results/decisions.json")
db = sqlite3.connect("results/results.db")


def ai_scores(model, condition="as downloaded"):
    """{(set, file): (class, probability of 'AI')} on the development data."""
    rows = db.execute(
        "SELECT p.set_name, p.file, i.class, p.predicted, p.score FROM predictions p JOIN images i USING (set_name, file) "
        f"WHERE p.model=? AND p.condition=? AND p.set_name IN {DEV}", (model, condition)).fetchall()
    assert len(rows) == 541, f"{model}, {condition}: {len(rows)} development images instead of 541"
    return {(s, f): (c, score if predicted == "ai" else 1 - score) for s, f, c, predicted, score in rows}


def accuracy(scores, threshold):
    return 100 * sum((p >= threshold) == (c == "ai") for c, p in scores.values()) / len(scores)


def best_threshold(scores):
    """The threshold on the 'AI' score with the highest accuracy; among equals the one closest to 0.5.

    Candidates lie halfway between neighbouring scores, so that rounding cannot move an image across the threshold.
    """
    values = sorted({p for _, p in scores.values()})
    candidates = [0.5] + [(a + b) / 2 for a, b in zip(values, values[1:])]
    return max(candidates, key=lambda t: (accuracy(scores, t), -abs(t - 0.5)))


assert best_threshold({1: ("ai", 0.9), 2: ("real", 0.7), 3: ("real", 0.2)}) == 0.8


def setting():
    def numbers(model, heading):
        check = db.execute("SELECT accuracy FROM scores WHERE model=? AND set_name='check' AND condition='as downloaded'", (model,)).fetchone()[0]
        main = db.execute("SELECT accuracy FROM scores WHERE model=? AND set_name='main' AND condition='as downloaded'", (model,)).fetchone()[0]
        steps = db.execute("SELECT real_correct + ai_correct, real_images + ai_images FROM processing WHERE model=?", (model,)).fetchall()
        assert len(steps) == 6, f"the processing test of {heading} is missing"
        return {"development": main, "check": check, "processing_mean": round(100 * sum(a for a, _ in steps) / sum(n for _, n in steps), 2)}

    old, new = numbers("cnn_final", "setting 4"), numbers("cnn_s5_raw", "setting 5")
    rules = {"higher on the development test set": new["development"] > old["development"],
             "processing mean not lower": new["processing_mean"] >= old["processing_mean"],
             "check set not more than two points lower": new["check"] >= old["check"] - 2}
    # For the record, not part of the rule: the two settings once each has the threshold of rule 2 (see the ablation in the report).
    tuned = {name: threshold(model) for name, model in (("setting 4", "cnn_final"), ("setting 5", "cnn_s5_raw"))}
    return {"setting 4": old, "setting 5": new, "rules": rules, "adopted": all(rules.values()),
            "for comparison, with the threshold of rule 2": {
                name: {key: t[key] for key in ("threshold_on_ai_score", "development_accuracy", "development_accuracy_jpeg90")} for name, t in tuned.items()}}


def threshold(model):
    scores = ai_scores(model)
    t = best_threshold(scores)
    return {"model": model, "threshold_on_ai_score": round(t, 6), "threshold_on_real_score": round(1 - t, 6),
            # Added to the 'real' logit of the published model, so that its own answer uses this threshold.
            "shift_of_real_logit": round(math.log(t / (1 - t)), 6),
            "development_accuracy_at_0.5": round(accuracy(scores, 0.5), 2), "development_accuracy": round(accuracy(scores, t), 2),
            "development_accuracy_jpeg90": round(accuracy(ai_scores(model, "jpeg90"), t), 2)}


def fusion(cnn="cnn_s5", other="commfor"):
    parts = {c: (ai_scores(cnn, c), ai_scores(other, c)) for c in ("as downloaded", "jpeg90")}
    tried = {}
    for w in (0.25, 0.5, 0.75):
        mixed = {c: {k: (a[k][0], w * a[k][1] + (1 - w) * b[k][1]) for k in a} for c, (a, b) in parts.items()}
        t = best_threshold(mixed["as downloaded"])
        tried[w] = {"threshold_on_ai_score": round(t, 6), "development_accuracy": round(accuracy(mixed["as downloaded"], t), 2),
                    "development_accuracy_jpeg90": round(accuracy(mixed["jpeg90"], t), 2)}
    w = max(tried, key=lambda w: (tried[w]["development_accuracy"], tried[w]["development_accuracy_jpeg90"]))
    alone = {name: {"development_accuracy": round(accuracy(parts["as downloaded"][i], 0.5), 2),
                    "development_accuracy_jpeg90": round(accuracy(parts["jpeg90"][i], 0.5), 2)} for i, name in enumerate((cnn, other))}
    adopted = all(tried[w][key] > alone[name][key] for name in alone for key in ("development_accuracy", "development_accuracy_jpeg90"))
    return {"weight_of_cnn": w, **tried[w], "tried": tried, "parts_alone": alone, "adopted": adopted}


# ---------------------------------------------------------------- third round
DEV3, OLD, NEW = ("check", "main", "final", "chat", "wild"), ("check", "main", "final"), ("chat", "wild")
CANDIDATES = {"A": "r3_A_raw", "B": "r3_B_raw", "C": "r3_C_raw"}  # in the order of the last tie-break
# Decision 2 of the third round, as committed in PROTOCOL.md: the figures of the current CNN (setting 5 before its
# threshold) on the 1,213 development images, and the limits that follow from them.
BASELINE3 = {"model": "cnn_s5_raw", "threshold": 0.6594, "old": 83.35, "old_ai": 77.97, "new": 68.40, "jpeg": 76.67}
LIMITS3 = {"new": 73.40, "old": 81.35, "old_ai": 72.97, "jpeg": 74.67}
assert LIMITS3 == {"new": round(BASELINE3["new"] + 5, 2), "old": round(BASELINE3["old"] - 2, 2),
                   "old_ai": round(BASELINE3["old_ai"] - 5, 2), "jpeg": round(BASELINE3["jpeg"] - 2, 2)}


def dev3_scores(model, condition="as downloaded"):
    """{(set, file): (set, class, probability of 'AI')} on the 1,213 development images of the third round."""
    rows = db.execute(
        "SELECT p.set_name, p.file, i.class, p.predicted, p.score FROM predictions p JOIN images i USING (set_name, file) "
        f"WHERE p.model=? AND p.condition=? AND p.set_name IN {DEV3}", (model, condition)).fetchall()
    assert len(rows) == 1213, f"{model}, {condition}: {len(rows)} development images instead of 1,213"
    return {(s, f): (s, c, score if predicted == "ai" else 1 - score) for s, f, c, predicted, score in rows}


def share(scores, threshold, sets=DEV3, cls=None):
    """Percent of the images of the given sets (and class) classified correctly at the threshold."""
    chosen = [(c, p) for s, c, p in scores.values() if s in sets and (cls is None or c == cls)]
    return 100 * sum((p >= threshold) == (c == "ai") for c, p in chosen) / len(chosen)


def figures3(model, t=None):
    """Decision 1 (threshold on all 1,213 images by rule 2) and the figures of decision 2 at that threshold.
    Returns the rounded figures for the record and the exact ones, which the rules compare."""
    scores, jpeg = dev3_scores(model), dev3_scores(model, "jpeg90")
    t = best_threshold({k: (c, p) for k, (_, c, p) in scores.items()}) if t is None else t
    exact = {"old": share(scores, t, OLD), "old_ai": share(scores, t, OLD, "ai"), "new": share(scores, t, NEW), "jpeg": share(jpeg, t)}
    exact["mean_old_new"] = (exact["old"] + exact["new"]) / 2
    record = {"threshold_on_ai_score": round(t, 6), "threshold_on_real_score": round(1 - t, 6),
              "shift_of_real_logit": round(math.log(t / (1 - t)), 6), **{k: round(v, 2) for k, v in exact.items()}}
    return record, exact


def qualifies(f):
    """Decision 2: learnt the new kinds, has not forgotten the old ones, and keeps its accuracy after JPEG re-compression."""
    return f["new"] >= LIMITS3["new"] and f["old"] >= LIMITS3["old"] and f["old_ai"] >= LIMITS3["old_ai"] and f["jpeg"] >= LIMITS3["jpeg"]


def choose(figs):
    """Among the qualifying candidates the highest mean of old and new; ties: higher JPEG, then the order A, B, C.
    figs: {candidate: exact figures}, in the order A, B, C. None if no candidate qualifies."""
    order = list(figs)
    ok = [c for c in order if qualifies(figs[c])]
    return max(ok, key=lambda c: (figs[c]["mean_old_new"], figs[c]["jpeg"], -order.index(c))) if ok else None


def round3():
    base, exact = figures3(BASELINE3["model"])
    assert (round(base["threshold_on_ai_score"], 4), *(base[k] for k in ("old", "old_ai", "new", "jpeg"))) == \
        tuple(BASELINE3[k] for k in ("threshold", "old", "old_ai", "new", "jpeg")), f"the baseline differs from PROTOCOL.md: {base}"
    figs = {c: figures3(m) for c, m in CANDIDATES.items()}
    adopted = choose({c: exact for c, (_, exact) in figs.items()})
    return {"baseline": {"model": BASELINE3["model"], **base}, "limits": LIMITS3,
            "candidates": {c: {"model": CANDIDATES[c], **record, "qualifies": qualifies(exact)} for c, (record, exact) in figs.items()},
            "adopted": adopted, "adopted_model": CANDIDATES.get(adopted),
            # for the secondary comparison: the Community Forensics detector with its threshold chosen on the same images
            "commfor": figures3("commfor")[0]}


def pick_weight(tried):
    """Decision 4: the weight with the highest mean of old and new; ties: higher JPEG, then the smaller weight."""
    return max(tried, key=lambda w: (tried[w]["mean_old_new"], tried[w]["jpeg"], -w))


def round3_fusion(cnn="cnn_r3", other="commfor"):
    """Decision 4 of the third round: the adopted candidate with its threshold built in (cnn_r3), combined with Community
    Forensics as in rule 3, the weight chosen by the mean of old and new; adopted if both that mean and the JPEG
    figure are higher than those of each part, each at its own threshold of decision 1."""
    decided = json.loads(OUT.read_text())["round3"]
    assert decided["adopted"], "no candidate was adopted: the system of the second round is kept"
    parts = {c: (dev3_scores(cnn, c), dev3_scores(other, c)) for c in ("as downloaded", "jpeg90")}
    tried, exact = {}, {}
    for w in (0.25, 0.5, 0.75):
        mixed = {c: {k: (s, cl, w * a[k][2] + (1 - w) * b[k][2]) for k, (s, cl, _) in a.items()} for c, (a, b) in parts.items()}
        t = best_threshold({k: (cl, p) for k, (_, cl, p) in mixed["as downloaded"].items()})
        exact[w] = {"old": share(mixed["as downloaded"], t, OLD), "old_ai": share(mixed["as downloaded"], t, OLD, "ai"),
                    "new": share(mixed["as downloaded"], t, NEW), "jpeg": share(mixed["jpeg90"], t)}
        exact[w]["mean_old_new"] = (exact[w]["old"] + exact[w]["new"]) / 2
        tried[w] = {"threshold_on_ai_score": round(t, 6), **{k: round(v, 2) for k, v in exact[w].items()}}
    w = pick_weight(exact)
    alone = {decided["adopted_model"]: figures3(decided["adopted_model"])[1], other: figures3(other)[1]}
    adopted = all(exact[w][k] > part[k] for part in alone.values() for k in ("mean_old_new", "jpeg"))
    return {"cnn": cnn, "weight_of_cnn": w, **tried[w], "tried": tried,
            "parts_alone": {m: {k: round(v, 2) for k, v in f.items()} for m, f in alone.items()}, "adopted": adopted}


def check():
    recorded = json.loads(OUT.read_text())
    again = {"setting": setting, "threshold": lambda: {model: threshold(model) for model in recorded["threshold"]}, "fusion": fusion,
             "round3": round3, "round3_fusion": round3_fusion}
    for name, decision in recorded.items():
        assert json.loads(json.dumps(again[name]())) == decision, f"the decision '{name}' is not reproduced from the database"
    print("reproduced from the database:", ", ".join(recorded))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decision", choices=["setting", "threshold", "fusion", "round3", "round3_fusion", "check"])
    parser.add_argument("--model", default="cnn_s5_raw", help="for threshold: the run whose threshold is chosen")
    args = parser.parse_args()
    if args.decision == "check":
        raise SystemExit(check())
    decisions = json.loads(OUT.read_text()) if OUT.exists() else {}
    if args.decision == "threshold":
        decisions.setdefault("threshold", {})[args.model] = threshold(args.model)
        shown = decisions["threshold"][args.model]
    else:
        shown = decisions[args.decision] = {"setting": setting, "fusion": fusion, "round3": round3, "round3_fusion": round3_fusion}[args.decision]()
    OUT.write_text(json.dumps(decisions, indent=1))
    print(json.dumps(shown, indent=1))
