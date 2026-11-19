"""Check that no real photograph of the evaluation sets is also a training image.

  python check_overlap.py

Every real training image and every real photograph of the evaluation sets is reduced to a
256-bit difference hash, which survives resizing and re-compression. A test photograph (whole image
and central square) that differs from a training image in at most 24 bits is reported as a possible
copy and should then be looked at. Unrelated images differ in about 128 bits. A copy that was
cropped differently would be missed.
"""
import io
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

REAL_SHARDS = [117, 116, 163, 101]  # the real-image shards of the training data (see train.py)
SOURCES = [("OwensLab/CommunityForensics-Small", f"data/HFCF_small_{n}.parquet", "image_data") for n in REAL_SHARDS] + \
          [("Dataseeds/DataSeeds.AI-Sample-Dataset-DSD", f"data/train-{n:05d}-of-00008.parquet", "image") for n in range(8)]
LIMIT = 24


def dhash(image):
    pixels = list(image.convert("L").resize((17, 16), Image.LANCZOS).getdata())
    return sum(1 << (row * 16 + col) for row in range(16) for col in range(16) if pixels[row * 17 + col] > pixels[row * 17 + col + 1])


def central_square(image):
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def training_hashes():
    hashes = {}
    for repo, file, column in SOURCES:
        parquet = pq.ParquetFile(hf_hub_download(repo, file, repo_type="dataset"))
        for number, batch in enumerate(parquet.iter_batches(batch_size=64, columns=[column])):
            for i, value in enumerate(batch.column(0).to_pylist()):
                data = value["bytes"] if isinstance(value, dict) else value
                hashes[dhash(Image.open(io.BytesIO(data)))] = f"{file} row {number * 64 + i}"
    return hashes


if __name__ == "__main__":
    training = training_hashes()
    photos = sorted(p for folder in ("test", "check", "main", "final", "chat", "wild") for p in Path("data", folder, "real").glob("*.jpg"))
    found = 0
    for path in photos:
        image = Image.open(path)
        for candidate in {dhash(image), dhash(central_square(image))}:
            for known, where in training.items():
                if (candidate ^ known).bit_count() <= LIMIT:
                    found += 1
                    print(f"possible copy: {path} and {where}")
    print(f"compared {len(photos)} real photographs of the evaluation sets with {len(training)} different real training images: "
          f"{found} possible copies")
