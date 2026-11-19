"""Inspect the files of every detector and time it on this computer's processor.

  python analyse_models.py                # all detectors -> results/models.json (read by results_db.py)
  python analyse_models.py cnn_s5 fusion  # only these; the others keep their entries

For each detector: the files in its repository, the format and size of the weights, the number of
weights, the architecture and input size from its configuration, the licence and dates from the
Hugging Face Hub, how much its model card says, and the seconds it needs per image on a CPU.
"""
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from transformers import pipeline
from transformers.image_utils import load_image

MODELS = {  # name used in the result files: Hub id, or folder of a model that was not published
    "cnn_s5": "models/cnn_s5",          # setting 5 with its threshold built in: the published model
    "cnn_final": "models/convnext_robust",  # setting 4, the model published in the first round
    "cnn_setting3": "models/convnext_crop_photos",
    "cnn_setting2": "models/convnext_crop",
    "cnn_first": "models/convnext_resize",
    "commfor": "OwensLab/commfor-model-384",
    "ateeqq": "Ateeqq/ai-vs-human-image-detector",
    "umm_maybe": "umm-maybe/AI-image-detector",
    "vit_baseline": "Manish18/deepfake-detection-model",
    "siglip2_baseline": "prithivMLmods/Deep-Fake-Detector-Model",
}
TIMED = sorted(p for p in Path("data/main").glob("*/*") if p.suffix.lower() in (".jpg", ".png"))[::24]  # 16 images, both classes


def count_weights(path):
    if path.suffix == ".safetensors":
        with safe_open(path, "pt") as f:
            return sum(math.prod(f.get_slice(key).get_shape()) for key in f.keys())
    return sum(t.numel() for t in torch.load(path, map_location="cpu", weights_only=True).values())  # no code is run


def inspect(source):
    facts = {"source": source}
    if Path(source).is_dir():
        folder = Path(source)
        files = {p.name: p.stat().st_size for p in folder.iterdir() if p.is_file()}
        fetch = lambda name: folder / name
    else:
        info = HfApi().model_info(source, files_metadata=True)
        files = {s.rfilename: s.size for s in info.siblings}
        fetch = lambda name: Path(hf_hub_download(source, name))
        tags = info.tags or []
        card = Path(hf_hub_download(source, "README.md")).read_text(encoding="utf-8") if "README.md" in files else ""
        text = card.split("---", 2)[-1] if card.startswith("---") else card
        facts.update(
            hub_id=info.id, created=str(info.created_at)[:10], modified=str(info.last_modified)[:10],
            downloads_last_month=info.downloads, likes=info.likes,
            licence=next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None),
            base_model=next((t.split(":")[-1] for t in tags if t.startswith("base_model:")), None),
            card_words=len(text.split()), card_unfilled_fields=text.count("More Information Needed"))
    weights = [n for n in files if n.endswith((".safetensors", ".bin", ".pt", ".pth", ".ckpt"))]
    main = next((n for n in weights if n.endswith(".safetensors")), weights[0])
    config = json.loads(fetch("config.json").read_text())
    vision = config.get("vision_config") or config
    facts.update(
        files=sorted(files), weight_files={n: round(files[n] / 1e6, 1) for n in weights},
        safe_weights=main.endswith(".safetensors"),  # a .bin file is a Python pickle and can run code when loaded
        weights_million=round(count_weights(fetch(main)) / 1e6, 1),
        architecture=(config.get("architectures") or [f"ViT-{config.get('model_size', '?')} (timm)"])[0],
        model_type=config.get("model_type") or "timm vit",
        image_size=vision.get("image_size") or config.get("input_size"), patch_size=vision.get("patch_size"),
        labels=list((config.get("id2label") or {}).values()) or ["one output: generated"])
    if "preprocessor_config.json" in files:
        pre = json.loads(fetch("preprocessor_config.json").read_text())
        facts["resizes_input"] = bool(pre.get("do_resize", True))
        facts["preprocessor_size"] = pre.get("size")
    else:
        facts["resizes_input"] = True  # commfor.py: resize to 440, centre crop of 384
    return facts


def seconds_per_image(name, source):
    if name == "commfor":
        from commfor import classifier
        classify = classifier(source)
    else:
        pipe = pipeline("image-classification", model=source, device="cpu")
        classify = lambda image: pipe(image)
    images = [load_image(str(p)) for p in TIMED]
    classify(images[0])  # the first call loads libraries and is not timed
    seconds = []
    for image in images:
        start = time.perf_counter()
        classify(image)
        seconds.append(time.perf_counter() - start)
    return {"cpu_mean_s": round(statistics.mean(seconds), 3), "cpu_max_s": round(max(seconds), 3)}


if __name__ == "__main__":
    assert len(TIMED) == 16, "the timing needs the main test set in data/main"
    megapixels = [im.width * im.height / 1e6 for im in (load_image(str(p)) for p in TIMED)]
    target = Path("results/models.json")
    out = json.loads(target.read_text()) if target.exists() and sys.argv[1:] else {}
    out["_timing"] = {"images": len(TIMED), "mean_megapixels": round(statistics.mean(megapixels), 2),
                      "max_megapixels": round(max(megapixels), 2), "threads": torch.get_num_threads()}
    for name, source in MODELS.items():
        if sys.argv[1:] and name not in sys.argv[1:]:
            continue
        out[name] = {**inspect(source), **seconds_per_image(name, source)}
        print(name, json.dumps({k: v for k, v in out[name].items() if k != "files"})[:400])
    target.write_text(json.dumps(out, indent=1))
    print("wrote results/models.json")
