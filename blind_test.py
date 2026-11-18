"""Blind test of a chat model (for example Claude) on one of the evaluation sets.

  python blind_test.py prepare [--set main]
  python blind_test.py score ANSWERS.txt NAME --model "model name" [--set main]

prepare writes shuffled copies of the images of a set to data/blind_SET/ as img_001.jpg, img_002.jpg ...
Every copy is re-encoded as JPEG (longest side at most 1568 px), so the file name, folder,
format and metadata no longer reveal the class. The key stays in data/blind_SET_key.csv.
The default set is the pilot test set (data/test); its results go to results/, those of any
other set to results/SET/.

The model is shown each image with this question, the same one as in the first evaluation:
  "Is this image a real photograph taken by a camera, or is it AI-generated / synthetic?
   Answer with only one word: Real or AI-Generated."

ANSWERS.txt holds one line per image, such as "img_007: Real" or "img_007: AI-Generated".
"""
import argparse
import csv
import random
import re
from pathlib import Path

import evaluate
from evaluate import IMAGE_TYPES, save


def prepare(source, blind, key_file):
    from PIL import Image

    files = sorted(p for p in source.glob("*/*") if p.suffix.lower() in IMAGE_TYPES)
    random.Random(48).shuffle(files)
    blind.mkdir(parents=True, exist_ok=True)
    key = []
    for number, path in enumerate(files, 1):
        image = Image.open(path).convert("RGB")
        image.thumbnail((1568, 1568))
        image.save(blind / f"img_{number:03d}.jpg", quality=92)  # a new file: no metadata is copied
        key.append([f"img_{number:03d}", path.parent.name, path.name])
    with open(key_file, "w", newline="") as f:
        csv.writer(f).writerows(key)
    print(f"wrote {len(key)} images to {blind}")


def score(answers_file, name, model, key_file):
    text = Path(answers_file).read_text(encoding="utf-8")
    found = re.findall(r"img_0*(\d+)\W+(real|ai-generated)\b", text, re.I)
    answers = {int(number): answer.lower() for number, answer in found}
    key = [(int(image.split("_")[1]), truth, file) for image, truth, file in csv.reader(open(key_file))]
    missing = [number for number, _, _ in key if number not in answers]
    if missing:
        raise SystemExit(f"No answer found for images {missing}")
    save(name, model, [[file, truth, "real" if answers[number] == "real" else "ai", ""] for number, truth, file in key])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "score"])
    parser.add_argument("answers", nargs="?")
    parser.add_argument("name", nargs="?")
    parser.add_argument("--model", default="chat model")
    parser.add_argument("--set", default="test", help="name of the folder under data/ that holds the images")
    args = parser.parse_args()
    blind, key_file = Path(f"data/blind_{args.set}"), Path(f"data/blind_{args.set}_key.csv")
    if args.set != "test":
        evaluate.RESULTS = Path("results") / args.set
    if args.command == "prepare":
        prepare(Path("data") / args.set, blind, key_file)
    elif args.answers and args.name:
        score(args.answers, args.name, args.model, key_file)
    else:
        parser.error("score needs ANSWERS.txt and NAME")
