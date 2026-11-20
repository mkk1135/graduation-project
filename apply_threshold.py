"""Build the decision threshold that was chosen on development data into a copy of a trained model.

  python apply_threshold.py TRAINED_FOLDER RUN_NAME OUT_FOLDER
  python apply_threshold.py checkpoints/s5_seed42/checkpoint-3087 cnn_s5_raw models/cnn_s5

The threshold of RUN_NAME is read from results/decisions.json (written by `python decide.py threshold --model RUN_NAME`).
A constant is added to the 'real' output of the network, so that its own answer, "the class with the higher
score", is the answer at the chosen threshold. Nothing else in the model changes.
"""
import json
import sys
from pathlib import Path

import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

if __name__ == "__main__":
    source, run, out = sys.argv[1:4]
    decisions = json.loads(Path("results/decisions.json").read_text())
    chosen = decisions["threshold"].get(run) or next(  # a candidate of the third round: its threshold of decision 1
        c for c in decisions["round3"]["candidates"].values() if c["model"] == run)
    model = AutoModelForImageClassification.from_pretrained(source).eval()
    ai, real = model.config.label2id["ai"], model.config.label2id["real"]
    x = torch.rand(4, 3, 256, 256) * 4 - 2
    with torch.no_grad():
        before = model(pixel_values=x).logits.softmax(-1)[:, ai]
        model.classifier.bias[real] += chosen["shift_of_real_logit"]
        after = model(pixel_values=x).logits.softmax(-1)[:, ai]
    # Self-check: the new default answer is the old answer at the chosen threshold.
    assert ((before >= chosen["threshold_on_ai_score"]) == (after >= 0.5)).all(), (before, after)
    model.save_pretrained(out)
    AutoImageProcessor.from_pretrained(source).save_pretrained(out)
    print(f"wrote {out}: threshold {chosen['threshold_on_ai_score']} on the 'AI' score of {run} is now the model's default")
