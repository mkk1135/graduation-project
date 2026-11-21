"""The combined detector: the fine-tuned CNN of this project together with the published Community Forensics detector.

Its score is a weighted mean of the two detectors' probabilities that an image is generated. The weight and the
threshold were chosen on development data by fixed rules (PROTOCOL.md, decide.py) and are read from
results/decisions.json. In evaluate.py the model id "fusion" selects it; "fusion:FOLDER" uses a local copy of the CNN.
"""
import json
from pathlib import Path

from evaluate import fit

CNN, OTHER = "mkk1135/ai-image-detector", "OwensLab/commfor-model-384"


def answer(p, threshold):
    """("ai" or "real", score) for the combined probability p that the image is generated.
    Rescaled so that the chosen threshold lies at 0.5: a score of 0.5 is an undecided image, as for a single detector."""
    p = 0.5 * p / threshold if p < threshold else 0.5 + 0.5 * (p - threshold) / (1 - threshold)
    return ("ai", p) if p >= 0.5 else ("real", 1 - p)


def classifier(device="cpu", cnn=CNN, decision="fusion"):
    """Returns a function: PIL image -> ("real" or "ai", score). The score is the probability of the class given.
    decision: the record in results/decisions.json that holds the weight and threshold ("round3_fusion" for the third round;
    "round3_seeds_fusion/seed1" for a record inside another)."""
    from transformers import pipeline  # imported here, so that answer() can be tested without PyTorch

    import commfor

    chosen = json.loads(Path(__file__).with_name("results").joinpath("decisions.json").read_text())
    for key in decision.split("/"):
        chosen = chosen[key]
    weight, threshold = chosen["weight_of_cnn"], chosen["threshold_on_ai_score"]
    own, other = pipeline("image-classification", model=cnn, device=device), commfor.classifier(OTHER, device)

    def classify(image):
        p_own = next(r["score"] for r in own(fit(image)) if r["label"] == "ai")  # the CNN works at original resolution
        kind, score = other(image)
        return answer(weight * p_own + (1 - weight) * (score if kind == "ai" else 1 - score), threshold)

    return classify
