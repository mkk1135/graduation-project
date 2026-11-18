"""Build the 48-image test set (26 real photographs, 22 AI-generated images) or the check set.

  python make_test_set.py            # test set  -> data/test,  results/test_set.csv
  python make_test_set.py --check    # check set -> data/check, results/check/check_set.csv

Real: Wikimedia Commons. From each of six photo categories, the most recently added
JPEG files that carry camera EXIF data and are at least 2000 px wide, one per uploader.

AI-generated: the Community Forensics evaluation set (Park and Owens, CVPR 2025). The
first two images of each of 11 commercial generators, none of which is in the training data.

The check set is drawn the same way from the same sources (15 photographs per category,
8 images per generator) and shares no image and no photographer with the test set. It is
used to choose between training settings, so that the test set is only used once.

The CSV lists what was drawn (source, author or generator, licence, URL). The Commons draw
depends on the day it is run, so an existing list is never overwritten.
"""
import argparse
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
CATEGORIES = {  # Commons category: number of photographs in the test set
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
CHECK_SHARDS = [47, 139, 175, 207, 282, 330, 341, 384]  # 300 MB more, for eight further images each
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


def real_photos(folder, per_category, exclude=None):
    """exclude: rows of the test set. Given for the check set, which may fall short in a small category."""
    skip_urls = {row["url"] for row in exclude or []}
    skip_authors = {row["author_or_generator"] for row in exclude or []}
    uploaders = set()
    for category, in_test in CATEGORIES.items():
        wanted = per_category or in_test
        newest = commons(action="query", list="categorymembers", cmtitle="Category:" + category, cmtype="file",
                         cmsort="timestamp", cmdir="desc", cmlimit="50" if exclude is None else "200")
        titles = [m["title"] for m in newest["query"]["categorymembers"]]
        info = {}
        for start in range(0, len(titles), 50):  # the API takes 50 titles at a time
            pages = commons(action="query", titles="|".join(titles[start:start + 50]), prop="imageinfo",
                            iiprop="url|size|mime|user|commonmetadata|extmetadata", iiurlwidth="1920")
            info.update({p["title"]: p["imageinfo"][0] for p in pages["query"]["pages"] if p.get("imageinfo")})
        count = 0
        for title in titles:
            i = info.get(title)
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
            count += 1
            name = f"{category.split()[-1]}_{count}.jpg"
            (folder / name).write_bytes(fetch(i["thumburl"]))  # the 1920 px wide rendition
            time.sleep(5)  # be polite to the Commons servers
            yield [f"real/{name}", "real", "Wikimedia Commons: " + category, author,
                   meta.get("LicenseShortName", {}).get("value", ""), i["descriptionurl"]]
        if count < wanted and exclude is None:
            raise SystemExit(f"Only {count} of {wanted} suitable photographs found in {category}")


def ai_images(folder, shards, per_generator, exclude=None):
    skip_urls = {row["url"] for row in exclude or []}
    taken = collections.Counter()
    for shard in shards:
        file = f"data/CompEval-{shard:05d}-of-00413.parquet"
        table = pq.read_table(hf_hub_download(EVAL, file, repo_type="dataset"),
                              columns=["image_name", "image_data", "model_name", "label"])
        for row in table.to_pylist():
            generator = row["model_name"]
            url = f"https://huggingface.co/datasets/{EVAL}/blob/main/{file} ({row['image_name']})"
            if row["label"] != 1 or generator not in GENERATORS or url in skip_urls:  # label 1 = generated
                continue
            if taken[generator] < per_generator:
                taken[generator] += 1
                name = f"{generator}_{taken[generator]}{Path(row['image_name']).suffix}"
                (folder / name).write_bytes(row["image_data"])
                yield [f"ai/{name}", "ai", "Community Forensics-Eval", generator, "CC BY-NC-SA 4.0", url]
    missing = [g for g in GENERATORS if taken[g] < per_generator]
    if missing:
        raise SystemExit(f"Fewer than {per_generator} images found for {missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="build the check set instead of the test set")
    check = parser.parse_args().check
    root, listing = (Path("data/check"), Path("results/check/check_set.csv")) if check else (Path("data/test"), Path("results/test_set.csv"))
    if listing.exists():
        raise SystemExit(f"{listing} already exists. Delete it only if you really want to draw a new set.")
    for sub in ("real", "ai"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    if check:
        test = list(csv.DictReader(open("results/test_set.csv", encoding="utf-8")))
        rows = [*real_photos(root / "real", 15, exclude=test),
                *ai_images(root / "ai", EVAL_SHARDS + CHECK_SHARDS, 8, exclude=test)]
    else:
        rows = [*real_photos(root / "real", None), *ai_images(root / "ai", EVAL_SHARDS, 2)]
        counts = collections.Counter(row[1] for row in rows)
        assert counts == {"real": 26, "ai": 22}, counts
    listing.parent.mkdir(parents=True, exist_ok=True)
    with open(listing, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([["file", "class", "source", "author_or_generator", "licence", "url"], *rows])
    print(f"wrote {len(rows)} images to {root} and {listing}")
