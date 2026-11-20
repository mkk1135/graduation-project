"""Score a detector again with the decision threshold that was chosen for it on development data.

  python rethreshold.py MODEL_NAME RESULTS_FOLDER [RESULTS_FOLDER ...]
  python rethreshold.py commfor results/final results/final/jpeg90 results/chat
  python rethreshold.py --round3 r3_A_raw results/final3 results/final3/jpeg90   # threshold of the third round -> r3_A

No image is classified again: the stored scores of MODEL_NAME in each folder are compared with the threshold
recorded in results/decisions.json (`python decide.py threshold --model MODEL_NAME`), and the results are
written next to them as MODEL_NAME_tuned.
"""
import csv
import json
import sys
from pathlib import Path

import evaluate

if __name__ == "__main__":
    round3 = sys.argv[1] == "--round3"  # the threshold chosen on the development data of the third round
    name, folders = sys.argv[1 + round3], sys.argv[2 + round3:]
    decisions = json.loads(Path("results/decisions.json").read_text())
    if round3:
        chosen = decisions["round3"]["commfor" if name == "commfor" else "candidates"]
        threshold = (chosen if name == "commfor" else next(c for c in chosen.values() if c["model"] == name))["threshold_on_ai_score"]
        out = name[:-len("_raw")] if name.endswith("_raw") else f"{name}_tuned3"
    else:
        threshold, out = decisions["threshold"][name]["threshold_on_ai_score"], f"{name}_tuned"
    for folder in folders:
        rows = []
        for r in csv.DictReader(open(Path(folder, f"{name}_predictions.csv"))):
            ai = float(r["score"]) if r["predicted"] == "ai" else 1 - float(r["score"])  # the stored score belongs to the stored answer
            rows.append([r["file"], r["truth"], "ai" if ai >= threshold else "real", round(ai if ai >= threshold else 1 - ai, 4)])
        evaluate.RESULTS = Path(folder)
        evaluate.save(out, f"{name}, threshold {threshold:.4f} on the 'AI' score (chosen on development data)", rows)
