"""Score a detector again with the decision threshold that was chosen for it on development data.

  python rethreshold.py MODEL_NAME RESULTS_FOLDER [RESULTS_FOLDER ...]
  python rethreshold.py commfor results/final results/final/jpeg90 results/chat

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
    name, folders = sys.argv[1], sys.argv[2:]
    threshold = json.loads(Path("results/decisions.json").read_text())["threshold"][name]["threshold_on_ai_score"]
    for folder in folders:
        rows = []
        for r in csv.DictReader(open(Path(folder, f"{name}_predictions.csv"))):
            ai = float(r["score"]) if r["predicted"] == "ai" else 1 - float(r["score"])  # the stored score belongs to the stored answer
            rows.append([r["file"], r["truth"], "ai" if ai >= threshold else "real", round(ai if ai >= threshold else 1 - ai, 4)])
        evaluate.RESULTS = Path(folder)
        evaluate.save(f"{name}_tuned", f"{name}, threshold {threshold:.4f} on the 'AI' score (chosen on development data)", rows)
