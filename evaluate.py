"""Evaluate an image classifier on folders of real and AI-generated images.

  python evaluate.py MODEL_ID --real data/main/real --ai data/main/ai --name short_name --out results/main
  python evaluate.py --chart   --out results/main      # bar chart of every model in that folder
  python evaluate.py --compare cnn_final --out results/main   # McNemar test of every model against one

MODEL_ID is an image-classification model on the Hugging Face Hub (or a local folder), or
OwensLab/commfor-model-384, or "fusion" (the combined detector, see fusion.py). "real" is the positive class.
Set DETECTOR_DEVICE=cuda to use a GPU.
"""
import argparse
import csv
import io
import json
import math
import os
from pathlib import Path

from PIL import Image

RESULTS = Path("results")
DEVICE = os.environ.get("DETECTOR_DEVICE", "cpu")  # set DETECTOR_DEVICE=cuda to run the detectors on a GPU
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def metrics(tp, fn, fp, tn):
    """tp = real called real, fn = real called AI, fp = AI called real, tn = AI called AI."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": round(100 * (tp + tn) / (tp + fn + fp + tn), 2),
        "precision": round(100 * precision, 2),
        "recall": round(100 * recall, 2),
        "f1": round(100 * f1, 2),
    }


def wilson(correct, n, z=1.96):
    """95% Wilson score interval for a proportion, in percent."""
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(100 * (centre - half), 1), round(100 * (centre + half), 1)]


def mcnemar(b, c):
    """Exact two-sided McNemar test. b, c = images that only the first / only the second model got right."""
    n = b + c
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n) if n else 1.0


# Self-checks: the CNN row of the first report (4/26 real, 21/22 AI correct), and textbook values.
assert metrics(4, 22, 1, 21) == {
    "tp": 4, "fn": 22, "fp": 1, "tn": 21,
    "accuracy": 52.08, "precision": 80.0, "recall": 15.38, "f1": 25.81,
}
assert wilson(50, 100) == [40.4, 59.6] and mcnemar(0, 0) == 1.0 and round(mcnemar(10, 0), 5) == 0.00195


def to_class(label):
    # Covers REAL/FAKE, real/ai, human/artificial and hum/ai label sets.
    return "real" if any(word in label.lower() for word in ("real", "hum")) else "ai"


def load(model_id):
    """Returns a function: PIL image -> ("real" or "ai", score)."""
    if model_id.startswith("fusion"):  # "fusion", or "fusion:FOLDER" with a local copy of the CNN
        from fusion import CNN, classifier
        return classifier(DEVICE, model_id.partition(":")[2] or CNN)
    if model_id.startswith("OwensLab/commfor-model"):
        from commfor import classifier
        return classifier(model_id, DEVICE)
    from transformers import pipeline

    pipe = pipeline("image-classification", model=model_id, device=DEVICE)
    labels = set(pipe.model.config.id2label.values())
    if {to_class(label) for label in labels} != {"real", "ai"}:
        raise SystemExit(f"Cannot map the labels {labels} of {model_id} to real / ai")

    def classify(image):
        top = pipe(image)[0]
        return to_class(top["label"]), top["score"]

    return classify


def save(name, model, rows):
    """Write the metrics, per-image predictions and confusion matrix. rows: [file, truth, predicted, score]."""
    import matplotlib.pyplot as plt

    count = lambda truth, predicted: sum(r[1] == truth and r[2] == predicted for r in rows)
    result = {"model": model, "name": name, "images": len(rows),
              **metrics(count("real", "real"), count("real", "ai"), count("ai", "real"), count("ai", "ai"))}
    result["accuracy_ci95"] = wilson(result["tp"] + result["tn"], len(rows))

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2))
    with open(RESULTS / f"{name}_predictions.csv", "w", newline="") as f:
        csv.writer(f).writerows([["file", "truth", "predicted", "score"], *rows])

    matrix = [[result["tp"], result["fn"]], [result["fp"], result["tn"]]]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["Real", "AI-generated"])
    ax.set_yticks([0, 1], ["Real", "AI-generated"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(name)
    for i in range(2):
        for j in range(2):
            dark = matrix[i][j] > max(map(max, matrix)) / 2
            ax.text(j, i, matrix[i][j], ha="center", va="center", fontsize=16,
                    color="white" if dark else "black")
    fig.savefig(RESULTS / f"{name}_confusion.png", dpi=300, bbox_inches="tight")
    print(json.dumps(result, indent=2))


def evaluate(model_id, real_dir, ai_dir, name, jpeg=None, crop=None):
    from transformers.image_utils import load_image

    classify = load(model_id)
    rows = []
    for truth, folder in (("real", real_dir), ("ai", ai_dir)):
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_TYPES)
        if not files:
            raise SystemExit(f"No images found in {folder}")
        for path in files:
            image = load_image(str(path))
            if crop:  # what the browser demo analyses: the central part, at original resolution
                w, h = min(image.width, crop), min(image.height, crop)
                left, top = (image.width - w) // 2, (image.height - h) // 2
                image = image.crop((left, top, left + w, top + h))
            if jpeg:  # afterwards the file format and compression history are the same for both classes
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=jpeg)
                image = load_image(Image.open(buffer))
            predicted, score = classify(image)
            rows.append([path.name, truth, predicted, round(score, 4)])
    save(name, model_id, rows)


def chart():
    import matplotlib.pyplot as plt

    results = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    names = ["accuracy", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.8 / len(results)
    for i, r in enumerate(results):
        ax.bar([x + i * width for x in range(4)], [r[m] for m in names], width, label=r["name"])
    ax.set_xticks([x + 0.4 - width / 2 for x in range(4)], ["Accuracy", "Precision", "Recall", "F1-score"])
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(4, len(results)), frameon=False)  # above the bars
    fig.savefig(RESULTS / "comparison.png", dpi=300, bbox_inches="tight")
    print("wrote", RESULTS / "comparison.png")


def compare(reference):
    """McNemar test of every model in the results folder against the reference, on the same images."""
    def right(name):
        return {r["file"]: r["truth"] == r["predicted"] for r in csv.DictReader(open(RESULTS / f"{name}_predictions.csv"))}

    ref = right(reference)
    rows = [["model", "accuracy", "ci95_low", "ci95_high", f"only_{reference}_right", "only_this_model_right", "p_value"]]
    for path in sorted(RESULTS.glob("*.json")):
        result = json.loads(path.read_text())
        other = right(result["name"])
        assert other.keys() == ref.keys(), f"{result['name']} was scored on different images"
        b, c = sum(ref[f] and not other[f] for f in ref), sum(other[f] and not ref[f] for f in ref)
        rows.append([result["name"], result["accuracy"], *wilson(sum(other.values()), len(other)), b, c,
                     "" if result["name"] == reference else f"{mcnemar(b, c):.4g}"])
    with open(RESULTS / "significance.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    for row in rows:
        print(*row, sep=" | ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="model id")
    parser.add_argument("--real", help="folder of real photographs")
    parser.add_argument("--ai", help="folder of AI-generated images")
    parser.add_argument("--name", help="short name used for the output files")
    parser.add_argument("--chart", action="store_true", help="draw the combined comparison chart")
    parser.add_argument("--compare", metavar="NAME", help="test every model in the folder against this one")
    parser.add_argument("--out", default="results", help="folder for the output files")
    parser.add_argument("--jpeg", type=int, help="robustness check: first re-compress every image as JPEG at this quality")
    parser.add_argument("--crop", type=int, help="analyse only the central CROP x CROP pixels, as the browser demo does")
    args = parser.parse_args()
    RESULTS = Path(args.out)
    if args.chart:
        chart()
    elif args.compare:
        compare(args.compare)
    elif args.model and args.real and args.ai and args.name:
        evaluate(args.model, args.real, args.ai, args.name, args.jpeg, args.crop)
    else:
        parser.error("give MODEL --real DIR --ai DIR --name NAME, or --chart, or --compare NAME")
