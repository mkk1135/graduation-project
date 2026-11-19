"""Does a detector's error on real photographs depend on how strongly they are JPEG-compressed?

  python jpeg_quality.py MODEL_NAME [MODEL_NAME ...]     # names as in results/main, for example cnn_final cnn_s5

For the Open Images photographs of the development test set (the small web photographs), the strength of the
JPEG compression is read from each file: the mean of its luminance quantisation table, where a low value
means light compression. The script reports how many lightly and how many strongly compressed photographs
each model recognises as real, and writes the counts to results/main/jpeg_quality.json.
"""
import csv
import json
import sys
from pathlib import Path

from PIL import Image

LIGHT = 6  # mean quantiser below this value: light compression (about JPEG quality 90 and above)
OUT = Path("results/main/jpeg_quality.json")

if __name__ == "__main__":
    strength = {}
    for path in sorted(Path("data/main/real").glob("openimages_*.jpg")):
        with Image.open(path) as image:
            table = image.quantization[0]
            strength[path.name] = sum(table) / len(table)
    report = json.loads(OUT.read_text()) if OUT.exists() else {}
    for model in sys.argv[1:]:
        right = {r["file"]: r["predicted"] == "real" for r in csv.DictReader(open(f"results/main/{model}_predictions.csv")) if r["file"] in strength}
        light, strong = [f for f in right if strength[f] < LIGHT], [f for f in right if strength[f] >= LIGHT]
        report[model] = {"light": [sum(right[f] for f in light), len(light)], "strong": [sum(right[f] for f in strong), len(strong)]}
        print(f"{model}: lightly compressed {report[model]['light'][0]} of {len(light)} recognised, strongly compressed {report[model]['strong'][0]} of {len(strong)}")
    OUT.write_text(json.dumps(report, indent=1))
