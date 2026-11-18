"""Build the 48-image test set: 26 real photographs and 22 AI-generated images.

Real: Wikimedia Commons. From each of six photo categories, the most recently added
JPEG files that carry camera EXIF data and are at least 2000 px wide, one per uploader.

AI-generated: the Community Forensics evaluation set (Park and Owens, CVPR 2025). The
first two images of each of 11 commercial generators, none of which is in the training data.

Writes the images to data/test/real and data/test/ai, and the list of what was drawn
(source, author or generator, licence, URL) to results/test_set.csv. The Commons draw
depends on the day it is run; test_set.csv records the exact images that were used.
"""
import collections
import csv
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

USER_AGENT = {"User-Agent": "ai-image-detection-student-project/0.1"}
CATEGORIES = {  # Commons category: number of photographs
    "Quality images of people": 5,
    "Quality images of streets": 5,
    "Quality images of landscapes": 4,
    "Quality images of buildings": 4,
    "Quality images of birds": 4,
    "Quality images of food": 4,
}
EVAL = "OwensLab/CommunityForensics-Eval"
# The smallest shards (298 MB in total) that together hold two images of every generator below.
EVAL_SHARDS = [38, 96, 99, 129, 136, 150, 169, 176, 189, 213, 237, 281, 294, 346, 383, 387, 393]
GENERATORS = ["MidjourneyV6_1", "MidjourneyV5_2", "Dalle3", "Dalle2", "Firefly_Image3", "Firefly_Image2",
              "Imagen3", "IdeogramV2", "IdeogramV1", "FLUX-dev", "FLUX-schnell"]


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


def real_photos(folder):
    uploaders = set()
    for category, wanted in CATEGORIES.items():
        newest = commons(action="query", list="categorymembers", cmtitle="Category:" + category, cmtype="file",
                         cmsort="timestamp", cmdir="desc", cmlimit="50")["query"]["categorymembers"]
        pages = commons(action="query", titles="|".join(m["title"] for m in newest), prop="imageinfo",
                        iiprop="url|size|mime|user|commonmetadata|extmetadata", iiurlwidth="1920")["query"]["pages"]
        info = {p["title"]: p["imageinfo"][0] for p in pages if p.get("imageinfo")}
        count = 0
        for member in newest:
            i = info.get(member["title"])
            if count == wanted or not i:
                continue
            has_camera = any(m["name"] == "Model" for m in i.get("commonmetadata") or [])
            if i["mime"] != "image/jpeg" or i["width"] < 2000 or not has_camera or i["user"] in uploaders:
                continue
            uploaders.add(i["user"])
            count += 1
            name = f"{category.split()[-1]}_{count}.jpg"
            (folder / name).write_bytes(fetch(i["thumburl"]))  # the 1920 px wide rendition
            time.sleep(5)  # be polite to the Commons servers
            meta = i.get("extmetadata", {})
            author = re.sub(r"<[^>]+>", "", meta.get("Artist", {}).get("value", i["user"])).strip()
            yield [f"real/{name}", "real", "Wikimedia Commons: " + category, author,
                   meta.get("LicenseShortName", {}).get("value", ""), i["descriptionurl"]]
        if count < wanted:
            raise SystemExit(f"Only {count} of {wanted} suitable photographs found in {category}")


def ai_images(folder):
    taken = collections.Counter()
    for shard in EVAL_SHARDS:
        file = f"data/CompEval-{shard:05d}-of-00413.parquet"
        table = pq.read_table(hf_hub_download(EVAL, file, repo_type="dataset"),
                              columns=["image_name", "image_data", "model_name", "label"])
        for row in table.to_pylist():
            generator = row["model_name"]
            if row["label"] == 1 and generator in GENERATORS and taken[generator] < 2:  # label 1 = generated
                taken[generator] += 1
                name = f"{generator}_{taken[generator]}{Path(row['image_name']).suffix}"
                (folder / name).write_bytes(row["image_data"])
                yield [f"ai/{name}", "ai", "Community Forensics-Eval", generator, "CC BY-NC-SA 4.0",
                       f"https://huggingface.co/datasets/{EVAL}/blob/main/{file} ({row['image_name']})"]
    missing = [g for g in GENERATORS if taken[g] < 2]
    if missing:
        raise SystemExit(f"Fewer than two images found for {missing}")


if __name__ == "__main__":
    root = Path("data/test")
    for sub in ("real", "ai"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rows = [*real_photos(root / "real"), *ai_images(root / "ai")]
    counts = collections.Counter(row[1] for row in rows)
    assert counts == {"real": 26, "ai": 22}, counts
    Path("results").mkdir(exist_ok=True)
    with open("results/test_set.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["file", "class", "source", "author_or_generator", "licence", "url"], *rows])
    print(f"wrote {len(rows)} images to {root} and results/test_set.csv")
