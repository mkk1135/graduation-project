"""Evaluate a Hugging Face image classifier on the real / AI test folders.

  python evaluate.py MODEL_ID --real "path/Real Images" --ai "path/AI Images" --name cnn_original
  python evaluate.py --chart        # combined bar chart from every results/*.json

"real" is the positive class, as in the first evaluation report.
"""
import argparse
import csv
import io
import json
from pathlib import Path

from PIL import Image

RESULTS = Path("results")
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


# Self-check against the CNN row of the first report (4/26 real, 21/22 AI correct).
assert metrics(4, 22, 1, 21) == {
    "tp": 4, "fn": 22, "fp": 1, "tn": 21,
    "accuracy": 52.08, "precision": 80.0, "recall": 15.38, "f1": 25.81,
}


def to_class(label):
    # Covers REAL/FAKE, Real/Fake and real/ai label sets.
    return "real" if "real" in label.lower() else "ai"


def save(name, model, rows):
    """Write the metrics, per-image predictions and confusion matrix. rows: [file, truth, predicted, score]."""
    import matplotlib.pyplot as plt

    count = lambda truth, predicted: sum(r[1] == truth and r[2] == predicted for r in rows)
    result = {"model": model, "name": name, "images": len(rows),
              **metrics(count("real", "real"), count("real", "ai"), count("ai", "real"), count("ai", "ai"))}

    RESULTS.mkdir(exist_ok=True)
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


def evaluate(model_id, real_dir, ai_dir, name, jpeg=None):
    from transformers import pipeline
    from transformers.image_utils import load_image

    classify = pipeline("image-classification", model=model_id)
    labels = set(classify.model.config.id2label.values())
    if {to_class(label) for label in labels} != {"real", "ai"}:
        raise SystemExit(f"Cannot map the labels {labels} of {model_id} to real / ai")

    rows = []
    for truth, folder in (("real", real_dir), ("ai", ai_dir)):
        files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_TYPES)
        if not files:
            raise SystemExit(f"No images found in {folder}")
        for path in files:
            image = load_image(str(path))
            if jpeg:  # afterwards the file format and compression history are the same for both classes
                buffer = io.BytesIO()
                image.save(buffer, "JPEG", quality=jpeg)
                image = load_image(Image.open(buffer))
            top = classify(image)[0]
            rows.append([path.name, truth, to_class(top["label"]), round(top["score"], 4)])
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
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(results), frameon=False)  # above the bars
    fig.savefig(RESULTS / "comparison.png", dpi=300, bbox_inches="tight")
    print("wrote", RESULTS / "comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?", help="Hugging Face model id")
    parser.add_argument("--real", help="folder of real photographs")
    parser.add_argument("--ai", help="folder of AI-generated images")
    parser.add_argument("--name", help="short name used for the output files")
    parser.add_argument("--chart", action="store_true", help="draw the combined comparison chart")
    parser.add_argument("--out", default="results", help="folder for the output files")
    parser.add_argument("--jpeg", type=int, help="robustness check: first re-compress every image as JPEG at this quality")
    args = parser.parse_args()
    RESULTS = Path(args.out)
    if args.chart:
        chart()
    elif args.model and args.real and args.ai and args.name:
        evaluate(args.model, args.real, args.ai, args.name, args.jpeg)
    else:
        parser.error("give MODEL --real DIR --ai DIR --name NAME, or --chart")
