"""Take the decisions of PROTOCOL.md from the results database and record them in results/decisions.json.

  python decide.py setting      # rule 1: does setting 5 replace setting 4?   (needs cnn_s5_raw on the development data)
  python decide.py threshold    # rule 2: decision threshold of a CNN          (needs cnn_s5_raw; --model for another run)
  python decide.py fusion       # rule 3: combination with Community Forensics (needs cnn_s5 with its threshold built in)

Only the check set and the development test set are read. Run `python results_db.py` first.
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
    return {"setting 4": old, "setting 5": new, "rules": rules, "adopted": all(rules.values())}


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decision", choices=["setting", "threshold", "fusion"])
    parser.add_argument("--model", default="cnn_s5_raw", help="for threshold: the run whose threshold is chosen")
    args = parser.parse_args()
    decisions = json.loads(OUT.read_text()) if OUT.exists() else {}
    if args.decision == "threshold":
        decisions.setdefault("threshold", {})[args.model] = threshold(args.model)
        shown = decisions["threshold"][args.model]
    else:
        shown = decisions[args.decision] = setting() if args.decision == "setting" else fusion()
    OUT.write_text(json.dumps(decisions, indent=1))
    print(json.dumps(shown, indent=1))
