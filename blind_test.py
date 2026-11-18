"""Blind test of a chat model (for example Claude) on the test set.

  python blind_test.py prepare
  python blind_test.py score ANSWERS.txt NAME --model "model name"

prepare writes shuffled copies of the 48 test images to data/blind/ as img_01.jpg ... img_48.jpg.
Every copy is re-encoded as JPEG (longest side at most 1568 px), so the file name, folder,
format and metadata no longer reveal the class. The key stays in data/blind_key.csv.

The model is shown each image with this question, the same one as in the first evaluation:
  "Is this image a real photograph taken by a camera, or is it AI-generated / synthetic?
   Answer with only one word: Real or AI-Generated."

ANSWERS.txt holds one line per image, such as "img_07: Real" or "img_07: AI-Generated".
"""
import argparse
import csv
import random
import re
from pathlib import Path

from evaluate import IMAGE_TYPES, save

TEST, BLIND, KEY = Path("data/test"), Path("data/blind"), Path("data/blind_key.csv")


def prepare():
    from PIL import Image

    files = sorted(p for p in TEST.glob("*/*") if p.suffix.lower() in IMAGE_TYPES)
    random.Random(48).shuffle(files)
    BLIND.mkdir(parents=True, exist_ok=True)
    key = []
    for number, path in enumerate(files, 1):
        image = Image.open(path).convert("RGB")
        image.thumbnail((1568, 1568))
        image.save(BLIND / f"img_{number:02d}.jpg", quality=92)  # a new file: no metadata is copied
        key.append([f"img_{number:02d}", path.parent.name, path.name])
    with open(KEY, "w", newline="") as f:
        csv.writer(f).writerows(key)
    print(f"wrote {len(key)} images to {BLIND}")


def score(answers_file, name, model):
    text = Path(answers_file).read_text(encoding="utf-8")
    answers = {image: answer.lower() for image, answer in re.findall(r"(img_\d\d)\W+(real|ai-generated)\b", text, re.I)}
    key = list(csv.reader(open(KEY)))
    missing = [image for image, _, _ in key if image not in answers]
    if missing:
        raise SystemExit(f"No answer found for {missing}")
    save(name, model, [[file, truth, "real" if answers[image] == "real" else "ai", ""] for image, truth, file in key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "score"])
    parser.add_argument("answers", nargs="?")
    parser.add_argument("name", nargs="?")
    parser.add_argument("--model", default="chat model")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.answers and args.name:
        score(args.answers, args.name, args.model)
    else:
        parser.error("score needs ANSWERS.txt and NAME")
