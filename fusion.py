"""The combined detector: the fine-tuned CNN of this project together with the published Community Forensics detector.

Its score is a weighted mean of the two detectors' probabilities that an image is generated. The weight and the
threshold were chosen on development data by fixed rules (PROTOCOL.md, decide.py) and are read from
results/decisions.json. In evaluate.py the model id "fusion" selects it; "fusion:FOLDER" uses a local copy of the CNN.
"""
import json
from pathlib import Path

from transformers import pipeline

import commfor

CNN, OTHER = "mkk1135/ai-image-detector", "OwensLab/commfor-model-384"


def classifier(device="cpu", cnn=CNN):
    """Returns a function: PIL image -> ("real" or "ai", score). The score is the probability of the class given."""
    chosen = json.loads(Path(__file__).with_name("results").joinpath("decisions.json").read_text())["fusion"]
    weight, threshold = chosen["weight_of_cnn"], chosen["threshold_on_ai_score"]
    own, other = pipeline("image-classification", model=cnn, device=device), commfor.classifier(OTHER, device)

    def classify(image):
        p_own = next(r["score"] for r in own(image) if r["label"] == "ai")
        kind, score = other(image)
        p = weight * p_own + (1 - weight) * (score if kind == "ai" else 1 - score)
        # Rescaled so that the chosen threshold lies at 0.5: a score of 0.5 is an undecided image, as for a single detector.
        p = 0.5 * p / threshold if p < threshold else 0.5 + 0.5 * (p - threshold) / (1 - threshold)
        return ("ai", p) if p >= 0.5 else ("real", 1 - p)

    return classify
