"""Which image-processing steps change a detector's answers on the check set?

  python diagnose.py MODEL_ID [MODEL_ID ...]

Every check-set image is classified as downloaded and after one processing step (blur,
sharpening, downscaling, added noise). A detector that reacts to how clean or sharp an
image is, rather than to how it was made, shows up as a large change in one column.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from transformers import pipeline

from evaluate import IMAGE_TYPES, to_class

STEPS = {
    "as downloaded": lambda im: im,
    "Gaussian blur, radius 0.6": lambda im: im.filter(ImageFilter.GaussianBlur(0.6)),
    "Gaussian blur, radius 1.0": lambda im: im.filter(ImageFilter.GaussianBlur(1.0)),
    "unsharp mask (sharpen)": lambda im: im.filter(ImageFilter.UnsharpMask(radius=1, percent=80, threshold=0)),
    "downscaled to half size": lambda im: im.resize((im.width // 2, im.height // 2), Image.LANCZOS),
    "Gaussian noise, sd 3": lambda im: Image.fromarray(np.clip(
        np.asarray(im) + np.random.default_rng(0).normal(0, 3, (im.height, im.width, 3)), 0, 255).astype("uint8")),
}


def load(folder):
    return [Image.open(p).convert("RGB") for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in IMAGE_TYPES]


if __name__ == "__main__":
    real, ai = load("data/check/real"), load("data/check/ai")
    for model in sys.argv[1:]:
        classify = pipeline("image-classification", model=model)
        print(f"\n{model}")
        for name, step in STEPS.items():
            right_real = sum(to_class(classify(step(im))[0]["label"]) == "real" for im in real)
            right_ai = sum(to_class(classify(step(im))[0]["label"]) == "ai" for im in ai)
            print(f"  {name:28s} real correct {right_real:2d}/{len(real)}   AI correct {right_ai:2d}/{len(ai)}")
