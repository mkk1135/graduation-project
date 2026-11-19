"""Build one of the five evaluation sets and record what was drawn.

  python make_test_set.py test     # 48 images:  pilot test set -> data/test,  results/test_set.csv
  python make_test_set.py check    # 157 images: check set      -> data/check, results/check/check_set.csv
  python make_test_set.py main     # 384 images: main test set  -> data/main,  results/main/main_set.csv
  python make_test_set.py final    # 384 images: held-out final test set -> data/final, results/final/final_set.csv
  python make_test_set.py chat     # 168 images: ChatGPT and Gemini set -> data/chat, results/chat/chat_set.csv

Real photographs
  Wikimedia Commons: from each listed category, going back from the most recently added file, the
    JPEG files that carry camera EXIF data and are at least 2000 px wide, one per uploader, at 1920 px width.
  Unsplash Lite (main set only): a seeded random sample of photographs with camera EXIF data, at 1920 px width.
  Open Images V7 validation set (main set only): a seeded random sample, as distributed (at most 1024 px).

AI-generated images
  The Community Forensics evaluation set (Park and Owens, CVPR 2025): the first images of each listed
  generator, in shard order. None of the generators is in the training data.

ChatGPT and Gemini set
  ChatGPT: the test split of OpenFake, version 2.1 (ComplexDataLab/OpenFake, arXiv:2509.09495): in file order, the
    first images made with GPT Image 2 and GPT Image 1.5, without frames taken from videos. They are images that
    users published on Civitai. The first real photographs from DOCCI and ImageNet come from the same split.
  Gemini: the first images of bitmind/nano-banana (Nano Banana, that is Gemini 2.5 Flash Image) and the first
    generated images of the test split of julienlucas/nanobanana-pro-dataset (Nano Banana Pro). Both are
    community datasets with short descriptions; OpenFake's own Nano Banana Pro rows are mostly video frames.

Each set excludes every image of the sets before it, and the photographers of earlier Commons draws.
The check set is for choosing between training settings; the test sets are only for the final scores.
The CSV lists what was drawn (source, author or generator, licence, URL). The Commons draw depends on
the day it is run, so an existing list is never overwritten.
"""
import argparse
import collections
import csv
import io
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from PIL import Image

USER_AGENT = {"User-Agent": "ai-image-detection-student-project/0.1"}
EVAL = "OwensLab/CommunityForensics-Eval"
COMMERCIAL = ["MidjourneyV6_1", "MidjourneyV5_2", "Dalle3", "Dalle2", "Firefly_Image3", "Firefly_Image2",
              "Imagen3", "IdeogramV2", "IdeogramV1", "FLUX-dev", "FLUX-schnell"]
OPEN_SOURCE = ["Hourglass", "DFGAN", "kandinsky_2_2", "stable_cascade", "GALIP", "LCM_lora_sdxl",
               "kvikontent_midjourney_v6", "LCM_lora_sdv15", "DeciDiffusionV2", "LCM_lora_ssd1b"]
# Shards of the evaluation set, chosen as the smallest ones that hold enough images of every generator.
TEST_SHARDS = [38, 96, 99, 129, 136, 150, 169, 176, 189, 213, 237, 281, 294, 346, 383, 387, 393]  # 298 MB
CHECK_SHARDS = TEST_SHARDS + [47, 139, 175, 207, 282, 330, 341, 384]                             # 300 MB more
MAIN_SHARDS = CHECK_SHARDS + [7, 21, 55, 71, 80, 214, 246, 311, 357, 381]                         # 735 MB more
FINAL_SHARDS = MAIN_SHARDS + [84, 100, 205, 308]                                                  # 158 MB more
SETS = {
    "test": dict(
        root="data/test", listing="results/test_set.csv", earlier=[],
        commons={"people": 5, "streets": 5, "landscapes": 4, "buildings": 4, "birds": 4, "food": 4},
        shards=TEST_SHARDS, generators={g: 2 for g in COMMERCIAL}),
    "check": dict(
        root="data/check", listing="results/check/check_set.csv", earlier=["test"], may_fall_short=True,
        commons={c: 15 for c in ("people", "streets", "landscapes", "buildings", "birds", "food")},
        shards=CHECK_SHARDS, generators={g: 8 for g in COMMERCIAL}),
    "main": dict(
        root="data/main", listing="results/main/main_set.csv", earlier=["test", "check"], may_fall_short=True,
        # 64 in all: small categories run out of new photographers, and "objects" supplies the rest
        commons={"people": 10, "landscapes": 10, "buildings": 10, "food": 10, "flowers": 8, "markets": 8, "objects": None},
        commons_total=64,
        unsplash=64, open_images=64,
        shards=MAIN_SHARDS, generators={**{g: 12 for g in COMMERCIAL}, **{g: 6 for g in OPEN_SOURCE}}),
    "final": dict(  # drawn like the main test set, from other images, photographers and categories
        root="data/final", listing="results/final/final_set.csv", earlier=["test", "check", "main"], may_fall_short=True,
        commons={"trees": 8, "sculptures": 8, "ships": 8, "castles": 8, "mountains": 8, "churches": 8, "vehicles": 8, "objects": None},
        commons_total=64, unsplash=64, open_images=64, seed=2027,
        shards=FINAL_SHARDS, generators={**{g: 12 for g in COMMERCIAL}, **{g: 6 for g in OPEN_SOURCE}}),
    "chat": dict(
        root="data/chat", listing="results/chat/chat_set.csv", earlier=[],
        openfake={"gpt-image-2": 24, "gpt-image-1.5": 24, "docci": 36, "imagenet": 36},
        hub=[("bitmind/nano-banana", "data/train-00000-of-00031.parquet", None, "nano-banana", 24),  # (dataset, file, label, name, images)
             ("julienlucas/nanobanana-pro-dataset", "data/test-00000-of-00001.parquet", 0, "nano-banana-pro", 24)]),  # label 0 = generated
}


def download(url, path):
    """Fetch a file unless it is already there (an interrupted build can be continued)."""
    if not path.exists():
        path.write_bytes(fetch(url))
        return True
    return False


def fetch(url):
    for attempt in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=USER_AGENT), timeout=120).read()
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 4:
                raise
            time.sleep(int(error.headers.get("Retry-After") or 60))  # rate-limited: wait as asked


def commons(**params):
    params.update(format="json", formatversion="2")
    return json.loads(fetch("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params)))


def commons_photos(folder, categories, earlier, may_fall_short, total=None):
    skip_urls = {row["url"] for row in earlier}
    skip_authors = {row["author_or_generator"] for row in earlier if row["source"].startswith("Wikimedia")}
    uploaders, drawn = set(), 0
    for category, wanted in categories.items():
        title = "Quality images of " + category
        wanted = total - drawn if wanted is None else wanted  # None: whatever is still needed to reach the total
        count, more = 0, {}
        while count < wanted and more is not None:  # newest first, 50 files at a time
            newest = commons(action="query", list="categorymembers", cmtitle="Category:" + title, cmtype="file",
                             cmsort="timestamp", cmdir="desc", cmlimit="50", **more)
            more = newest.get("continue") if earlier else None  # the pilot test set looked at the newest 50 only
            titles = [m["title"] for m in newest["query"]["categorymembers"]]
            pages = commons(action="query", titles="|".join(titles), prop="imageinfo",
                            iiprop="url|size|mime|user|commonmetadata|extmetadata", iiurlwidth="1920")
            info = {p["title"]: p["imageinfo"][0] for p in pages["query"]["pages"] if p.get("imageinfo")}
            for title_ in titles:
                i = info.get(title_)
                if count == wanted or not i:
                    continue
                meta = i.get("extmetadata", {})
                author = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", i["user"])).strip()
                has_camera = any(m["name"] == "Model" for m in i.get("commonmetadata") or [])
                if i["mime"] != "image/jpeg" or i["width"] < 2000 or not has_camera:
                    continue
                if i["user"] in uploaders or i["descriptionurl"] in skip_urls or author in skip_authors:
                    continue
                uploaders.add(i["user"])
                count, drawn = count + 1, drawn + 1
                name = f"{category}_{count}.jpg"
                if download(i["thumburl"], folder / name):  # the 1920 px wide rendition
                    time.sleep(5)  # be polite to the Commons servers
                yield [f"real/{name}", "real", "Wikimedia Commons: " + title, author,
                       meta.get("LicenseShortName", {}).get("value", ""), i["descriptionurl"]]
        if count < wanted and not may_fall_short:
            raise SystemExit(f"Only {count} of {wanted} suitable photographs found in {title}")


def unsplash_photos(folder, wanted, seed, earlier):
    """A seeded random sample of the Unsplash Lite dataset (about 25,000 photographs)."""
    fs = HfFileSystem()
    photos = []
    for path in sorted(fs.glob("datasets/1aurent/unsplash-lite/data/*.parquet")):
        with fs.open(path) as f:
            photos += pq.read_table(f, columns=["photo", "photographer", "exif"]).to_pylist()
    photos = sorted((p for p in photos if p["exif"]["camera_model"] not in (None, "nan") and p["photo"]["width"] >= 2000),
                    key=lambda p: p["photo"]["id"])
    skip = {row["url"] for row in earlier if row["source"].startswith("Unsplash")}
    drawn = [p for p in random.Random(seed).sample(photos, wanted + len(skip)) if p["photo"]["url"] not in skip][:wanted]
    for number, p in enumerate(drawn, 1):
        name = f"unsplash_{number}.jpg"
        if download(p["photo"]["image_url"] + "?w=1920&q=85&fm=jpg", folder / name):
            time.sleep(1)
        author = " ".join(x for x in (p["photographer"]["first_name"], p["photographer"]["last_name"]) if x and x != "nan")
        yield [f"real/{name}", "real", "Unsplash Lite", author, "Unsplash License", p["photo"]["url"]]


def open_images_photos(folder, wanted, seed, earlier):
    """A seeded random sample of the Open Images V7 validation set (Flickr photographs, at most 1024 px)."""
    listing = fetch("https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv")
    rows = [r for r in csv.DictReader(listing.decode("utf-8").splitlines()) if r["Rotation"] in ("", "0.0")]
    skip = {row["url"] for row in earlier if row["source"].startswith("Open Images")}
    drawn = random.Random(seed).sample(sorted(rows, key=lambda r: r["ImageID"]), wanted + len(skip))
    for number, r in enumerate([r for r in drawn if r["OriginalLandingURL"] not in skip][:wanted], 1):
        name = f"openimages_{number}.jpg"
        download(f"https://open-images-dataset.s3.amazonaws.com/validation/{r['ImageID']}.jpg", folder / name)
        licence = "CC " + r["License"].split("/licenses/")[-1].strip("/").replace("/", " ").upper()
        yield [f"real/{name}", "real", "Open Images V7 validation set", r["Author"], licence, r["OriginalLandingURL"]]


def ai_images(folder, shards, generators, earlier):
    skip_urls = {row["url"] for row in earlier}
    taken = collections.Counter()
    for shard in shards:
        file = f"data/CompEval-{shard:05d}-of-00413.parquet"
        table = pq.read_table(hf_hub_download(EVAL, file, repo_type="dataset"),
                              columns=["image_name", "image_data", "model_name", "label"])
        for row in table.to_pylist():
            generator = row["model_name"]
            url = f"https://huggingface.co/datasets/{EVAL}/blob/main/{file} ({row['image_name']})"
            if row["label"] != 1 or generator not in generators or url in skip_urls:  # label 1 = generated
                continue
            if taken[generator] < generators[generator]:
                taken[generator] += 1
                name = f"{generator}_{taken[generator]}{Path(row['image_name']).suffix}"
                (folder / name).write_bytes(row["image_data"])
                yield [f"ai/{name}", "ai", "Community Forensics-Eval", generator, "CC BY-NC-SA 4.0", url]
    missing = [g for g in generators if taken[g] < generators[g]]
    if missing:
        raise SystemExit(f"Too few images found for {missing}")


def extension(data):
    return {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}[Image.open(io.BytesIO(data)).format]


def hub_images(folder, repo, file, label, generator, wanted):
    """The first images of one parquet file of a dataset on the Hub (those with the given label, if one is given)."""
    sha, count = HfApi().dataset_info(repo).sha, 0  # the listing names the exact version that was read
    with HfFileSystem().open(f"datasets/{repo}@{sha}/{file}") as f:
        parquet = pq.ParquetFile(f)
        for group in range(parquet.num_row_groups):
            if label is not None and label not in parquet.read_row_group(group, columns=["label"]).column("label").to_pylist():
                continue  # nothing wanted in this group: its images are not downloaded
            for number, row in enumerate(parquet.read_row_group(group).to_pylist()):
                if count < wanted and (label is None or row["label"] == label):
                    count += 1
                    name = f"{generator}_{count}{extension(row['image']['bytes'])}"
                    (folder / name).write_bytes(row["image"]["bytes"])
                    yield [f"ai/{name}", "ai", f"{repo} on the Hugging Face Hub", generator, "MIT",
                           f"https://huggingface.co/datasets/{repo}/blob/{sha}/{file} (row group {group}, row {number})"]
            if count == wanted:
                return
    raise SystemExit(f"Only {count} of {wanted} images found in {repo}")


def openfake_images(root, wanted):
    """The first images of each listed generator or photo source in the test split of OpenFake, in file order."""
    fs, taken = HfFileSystem(), collections.Counter()
    for path in sorted(fs.glob("datasets/ComplexDataLab/OpenFake@v2.1/core/test-*.parquet")):
        with fs.open(path) as f:
            parquet = pq.ParquetFile(f)
            for group in range(parquet.num_row_groups):
                models = parquet.read_row_group(group, columns=["model"]).column("model").to_pylist()
                if not any(taken[m] < wanted.get(m, 0) for m in models):
                    continue  # nothing needed from this group: its images (about 100 MB) are not downloaded
                for row in parquet.read_row_group(group, columns=["image", "label", "model", "type", "row_id"]).to_pylist():
                    model = row["model"]
                    if taken[model] >= wanted.get(model, 0) or row["type"] == "video":  # no frames taken from videos
                        continue
                    taken[model] += 1
                    kind = "ai" if row["label"] == "fake" else "real"
                    name = f"{model}_{taken[model]}{extension(row['image']['bytes'])}"
                    (root / kind / name).write_bytes(row["image"]["bytes"])
                    yield [f"{kind}/{name}", kind, "OpenFake v2.1, test split", model, "CC BY-NC 4.0",
                           f"https://huggingface.co/datasets/ComplexDataLab/OpenFake/blob/v2.1/{path.split('/', 3)[3]} (row {row['row_id']})"]
                print(f"  {path.rsplit('/', 1)[1]} group {group}: {dict(taken)}", flush=True)
                if all(taken[m] >= n for m, n in wanted.items()):
                    return
    raise SystemExit(f"Too few images found: {dict(taken)}")


def listed(name):
    return list(csv.DictReader(open(SETS[name]["listing"], encoding="utf-8")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("set", choices=SETS)
    spec = SETS[parser.parse_args().set]
    root, listing = Path(spec["root"]), Path(spec["listing"])
    if listing.exists():
        raise SystemExit(f"{listing} already exists. Delete it only if you really want to draw a new set.")
    for sub in ("real", "ai"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    earlier = [row for name in spec["earlier"] for row in listed(name)]
    rows = [*(ai_images(root / "ai", spec["shards"], spec["generators"], earlier) if "generators" in spec else []),
            *(openfake_images(root, spec["openfake"]) if "openfake" in spec else []),
            *(row for source in spec.get("hub", []) for row in hub_images(root / "ai", *source)),
            *(unsplash_photos(root / "real", spec["unsplash"], spec.get("seed", 2026), earlier) if "unsplash" in spec else []),
            *(open_images_photos(root / "real", spec["open_images"], spec.get("seed", 2026), earlier) if "open_images" in spec else []),
            *(commons_photos(root / "real", spec["commons"], earlier, spec.get("may_fall_short", False), spec.get("commons_total"))
              if "commons" in spec else [])]
    listing.parent.mkdir(parents=True, exist_ok=True)
    with open(listing, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["file", "class", "source", "author_or_generator", "licence", "url"], *rows])
    print(f"wrote {len(rows)} images to {root} and {listing}:", dict(collections.Counter(row[1] for row in rows)))
