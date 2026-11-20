"""Build the training data of the third round (PROTOCOL.md): new data only, from OpenFake v2.1.

  python make_train_set.py select     # the fixed selection, from the datasets' metadata (no images)
  python make_train_set.py download   # the selected images, read by HTTP range requests (can be continued)
  python make_train_set.py finish     # leave out images over 20 MB and every image that matches an evaluation image
  python make_train_set.py check      # compare every training image with every evaluation image again

Everything large is kept on drive D: (ROUND3_DIR, default D:/ai-detector-round3): the metadata in meta/, the
downloaded images in raw/ (deleted by `finish`), the training data in train_set/*.parquet (read by
`train.py --data round3`). The list of the training images is results/round3/train_set.csv, the images left out
with the reason results/round3/train_left_out.csv.

The metadata (meta/meta_*.parquet: file, row group, row, class, generator, type, subreddit ... of every image) is read
by `scan`, which downloads only the parquet footers and the small columns (core/train: about 140 MB, 45 minutes).

Images are read from the parquet files of the Hub without downloading whole files. In supplement_civitai and
supplement_reddit the image column of a row group (100 images) is one compressed block, so whole row groups are read;
in the core train split the image column is stored uncompressed in pages of 1 to 5 images, so the single images of
the API generators are read page by page and checked against the split's SHA-256.
"""
import argparse
import csv
import hashlib
import io
import os
import random
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_url
from PIL import Image

REPO, REV = "ComplexDataLab/OpenFake", "v2.1"  # tag v2.1 = commit 00b2f0ae5f7684c4b00f794f4c500076a9355d
R3 = Path(os.environ.get("ROUND3_DIR", "D:/ai-detector-round3"))
META, RAW, OUT = R3 / "meta", R3 / "raw", R3 / "train_set"
LISTING, LEFT_OUT = Path("results/round3/train_set.csv"), Path("results/round3/train_left_out.csv")
SEED = 2026
# OpenFake generators that are also generators of the Community Forensics evaluation set (kept unseen) and Seedream
EXCLUDE = {"dalle-3", "midjourney-6", "imagen-3.0-002", "flux.1-dev", "flux.1-schnell", "seedream-v4.0", "seedream-v4.5"}
CORE_GROUPS, CORE_OTHER = 42, 350
API = {"gpt-image-1": 245, "imagen-4.0": 140, "nano-banana-2": 140, "grok-imagine": 70, "flux-1.1-pro": 35,
       "ideogram-3.0": 35, "qwen-image": 35}
CIVITAI = {"ChatGPT Images": 290, "Google's Nano Banana": 160, "Grok Imagine": 140, "OpenAI's GPT-image-1": 30,
           "Google Imagen 4": 20, "Qwen Image (API)": 20, "Wan 2.7 Image": 20, "Ideogram 4": 20}
AI_SUBS = ["aiart", "aiartwork", "genaigallery", "ai_art", "midjourney", "pro_ai_art", "aigeneratedart", "aiimages"]
PHOTO_SUBS = ["itookapicture", "amateurphotography", "wildlifephotography", "photocritique", "cityporn", "earthporn",
              "analog", "streetphotography", "foodporn", "casual_photography", "portraitphotography"]
REDDIT_AI, PER_PHOTO_SUB = 750, 90
MAX_BYTES, DHASH_LIMIT = 20_000_000, 24
CONFIGS = {"core_train": "core/train-", "supp_civitai": "supplement/civitai/", "supp_reddit": "supplement/reddit/"}  # file prefixes
SCAN_COLUMNS = {"core_train": ["label", "model", "type"],
                "supp_civitai": ["label", "model", "type", "row_id", "image_sha256"],
                "supp_reddit": ["label", "model", "type", "row_id", "image_sha256", "subreddit"]}


# ---------------------------------------------------------------- reading parquet files of the Hub by range requests
class RangeFile(io.RawIOBase):
    """A read-only file object over HTTP range requests, for pyarrow."""

    def __init__(self, path):
        self.client = httpx.Client(follow_redirects=True, timeout=120)
        for attempt in range(6):
            try:
                r = self.client.head(hf_hub_url(REPO, path, repo_type="dataset", revision=REV))
                r.raise_for_status()
                break
            except httpx.HTTPError:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        self.url, self.size, self.pos = str(r.url), int(r.headers["Content-Length"]), 0  # the storage URL, resolved once

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def fetch(self, start, n):
        for attempt in range(6):
            try:
                r = self.client.get(self.url, headers={"Range": f"bytes={start}-{start + n - 1}"})
                r.raise_for_status()
                return r.content
            except httpx.HTTPError:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        data = self.fetch(self.pos, n) if n > 0 else b""
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def thrift_struct(data, i=0):
    """Minimal Thrift compact-protocol reader (for parquet page headers): ({field: value}, end position)."""
    def varint():
        nonlocal i
        shift = result = 0
        while True:
            x = data[i]
            i += 1
            result |= (x & 0x7F) << shift
            shift += 7
            if not x & 0x80:
                return result

    def value(t):
        nonlocal i
        if t in (1, 2):
            return t == 1
        if t == 3:
            i += 1
            return data[i - 1]
        if t in (4, 5, 6):
            n = varint()
            return (n >> 1) ^ -(n & 1)
        if t == 7:
            i += 8
            return None
        if t == 8:
            n = varint()
            i += n
            return data[i - n:i]
        if t in (9, 10):
            h = data[i]
            i += 1
            n = h >> 4 if h >> 4 != 15 else varint()
            return [value(h & 0x0F) for _ in range(n)]
        if t == 12:
            return struct_()
        raise ValueError(f"thrift type {t}")

    def struct_():
        nonlocal i
        out, last = {}, 0
        while True:
            h = data[i]
            i += 1
            if h == 0:
                return out
            delta, t = h >> 4, h & 0x0F
            last = last + delta if delta else value(5)
            out[last] = value(t)

    return struct_(), i


def single_images(path, group, rows):
    """{row: (image bytes, row_id)} for some rows of one row group of the core train split, read page by page."""
    f = RangeFile(path)
    pf = pq.ParquetFile(f)
    col = pf.metadata.row_group(group).column(0)
    assert col.path_in_schema == "image.bytes" and col.compression == "UNCOMPRESSED" and col.dictionary_page_offset is None
    small = pf.read_row_group(group, columns=["row_id", "image_sha256"]).to_pylist()
    pos, end, first, out, wanted = col.data_page_offset, col.data_page_offset + col.total_compressed_size, 0, {}, set(rows)
    while pos < end and len(out) < len(wanted):
        try:
            header, length = thrift_struct(f.fetch(pos, 256))
        except IndexError:  # a header with statistics is longer
            header, length = thrift_struct(f.fetch(pos, 65536))
        values, body_at, body_size = header[5][1], pos + length, header[3]  # data page: number of values; page size
        inside = [k for k in range(first, first + values) if k in wanted]
        if inside:
            body = f.fetch(body_at, body_size)
            p = 4 + struct.unpack("<I", body[:4])[0]  # skip the definition levels: every image is present
            for k in range(first, first + values):
                size = struct.unpack("<I", body[p:p + 4])[0]
                if k in wanted:
                    out[k] = (body[p + 4:p + 4 + size], small[k]["row_id"])
                    assert hashlib.sha256(out[k][0]).hexdigest() == small[k]["image_sha256"], (path, group, k)
                p += 4 + size
        pos, first = body_at + body_size, first + values
    assert len(out) == len(wanted), (path, group)
    return out


def whole_group(path, group, rows):
    """{row: (image bytes, row_id)} for some rows of one row group, read whole."""
    t = pq.ParquetFile(RangeFile(path), pre_buffer=True).read_row_group(group, columns=["image", "row_id", "image_sha256"]).to_pylist()
    out = {k: (t[k]["image"]["bytes"], t[k]["row_id"]) for k in rows}
    for k, (data, _) in out.items():
        assert hashlib.sha256(data).hexdigest() == t[k]["image_sha256"], (path, group, k)
    return out


# ---------------------------------------------------------------- metadata and selection
def scan(name):
    """Metadata of every image of one config, from the footers and small columns only."""
    files = sorted(f for f in __import__("huggingface_hub").HfApi().list_repo_files(REPO, repo_type="dataset", revision=REV)
                   if f.startswith(CONFIGS[name]) and f.endswith(".parquet"))

    def one(path):
        pf = pq.ParquetFile(RangeFile(path), pre_buffer=True)
        rows = []
        for g in range(pf.num_row_groups):
            rg = pf.metadata.row_group(g)
            for i, row in enumerate(pf.read_row_group(g, columns=SCAN_COLUMNS[name]).to_pylist()):
                rows.append({"file": path, "group": g, "index": i, "group_rows": rg.num_rows,
                             "group_image_bytes": rg.column(0).total_compressed_size, **{k: None if v is None else str(v) for k, v in row.items()}})
        return rows

    with ThreadPoolExecutor(16) as pool:
        rows = [r for part in pool.map(one, files) for r in part]
    pq.write_table(pa.Table.from_pylist(rows), META / f"meta_{name}.parquet")


def meta(name):
    if not (META / f"meta_{name}.parquet").exists():
        scan(name)
    return pq.read_table(META / f"meta_{name}.parquet").to_pandas().sort_values(["file", "group", "index"], ignore_index=True)


def densest(df, mask, target):
    """Row groups in order of the density (selected images per MB) of the masked rows, until `target` rows."""
    g = df.assign(w=mask.astype(int)).groupby(["file", "group"]).agg(w=("w", "sum"), mb=("group_image_bytes", "first")).reset_index()
    g = g[g.w > 0].assign(d=lambda x: x.w / x.mb).sort_values(["d", "file", "group"], ascending=[False, True, True])
    g = g[g.w.cumsum().shift(fill_value=0) < target]
    return list(zip(g.file, g.group))


def in_groups(df, groups):
    groups = set(groups)
    return pd.Series([k in groups for k in zip(df.file, df.group)], index=df.index)


def select():
    parts = []
    core = meta("core_train")
    nonvideo = core.type.str.lower() != "video"
    groups = sorted(set(zip(core.file, core.group)))
    random.Random(SEED).shuffle(groups)
    chosen = in_groups(core, groups[:CORE_GROUPS])
    photos = core[chosen & nonvideo & (core.label == "real")]
    parts.append(photos.assign(part=photos.model.map({"laion": "core photos LAION", "pexels": "core photos Pexels"}), how="group"))
    others = core[chosen & nonvideo & (core.label == "fake") & ~core.model.isin(EXCLUDE)]
    parts.append(others.loc[sorted(random.Random(SEED).sample(list(others.index), CORE_OTHER))].assign(part="core other generators", how="group"))
    rng, pool = random.Random(SEED), core[~chosen & nonvideo]
    for model, n in API.items():
        rows = pool[pool.model == model]
        parts.append(rows.loc[sorted(rng.sample(list(rows.index), n))].assign(part="core API generators", how="single"))

    cv = meta("supp_civitai")
    usable = (cv.type != "civitai_video_frame") & cv.model.isin(list(CIVITAI))
    groups = [g for m, n in CIVITAI.items() for g in densest(cv, usable & (cv.model == m), n)]
    rows = cv[usable & in_groups(cv, groups)]
    parts.append(pd.concat([rows[rows.model == m].head(n) for m, n in CIVITAI.items()]).assign(part="Civitai API generators", how="group"))

    sr = meta("supp_reddit")
    sub, image = sr.subreddit.str.lower(), sr.type != "video_frame"
    ai = image & sub.isin(AI_SUBS)
    parts.append(sr[ai & in_groups(sr, densest(sr, ai, REDDIT_AI))].head(REDDIT_AI).assign(part="Reddit AI subreddits", how="group"))
    groups = [g for s in PHOTO_SUBS for g in densest(sr, image & (sub == s), PER_PHOTO_SUB)]
    rows = sr[image & sub.isin(PHOTO_SUBS) & in_groups(sr, groups)].assign(sub=sub)
    parts.append(pd.concat([rows[rows["sub"] == s].head(PER_PHOTO_SUB) for s in PHOTO_SUBS]).assign(part="Reddit photo subreddits", how="group"))

    sel = pd.concat(parts, ignore_index=True)
    sel["label"] = (sel.label == "fake").astype(int)  # 1 = generated, as in the training data of the earlier rounds
    sel["generator"] = sel.subreddit.where(sel.part.str.startswith("Reddit"), sel.model) if "subreddit" in sel else sel.model
    sel = sel[["part", "how", "file", "group", "index", "label", "generator", "group_image_bytes"]]
    assert not sel.duplicated(["file", "group", "index"]).any()
    pq.write_table(pa.Table.from_pandas(sel, preserve_index=False), META / "selection.parquet")
    print(sel.groupby(["part", "label"]).size().to_string())
    print("images", len(sel), "| row groups read whole:", sel[sel.how == "group"].groupby(["file", "group"]).ngroups,
          f"({sel[sel.how == 'group'].groupby(['file', 'group']).group_image_bytes.first().sum() / 1e9:.1f} GB)")


# ---------------------------------------------------------------- download
def unit_name(how, path, group):
    return hashlib.sha1(f"{how}|{path}|{group}".encode()).hexdigest()[:16] + ".parquet"


def fetch_unit(how, path, group, rows):
    """Download the wanted rows of one row group and keep them in raw/ (skipped if done before)."""
    target = RAW / unit_name(how, path, group)
    if target.exists():
        return 0
    got = (single_images if how == "single" else whole_group)(path, group, list(rows.index_in_group))
    table = pa.Table.from_pylist([{"file": path, "group": group, "index": k, "row_id": got[k][1], "image_data": got[k][0]}
                                  for k in rows.index_in_group])
    pq.write_table(table, target.with_suffix(".tmp"))
    target.with_suffix(".tmp").replace(target)
    return sum(len(got[k][0]) for k in got)


def download(threads):
    RAW.mkdir(parents=True, exist_ok=True)
    sel = pq.read_table(META / "selection.parquet").to_pandas().rename(columns={"index": "index_in_group"})
    units = list(sel.groupby(["how", "file", "group"]))
    units.sort(key=lambda u: u[0][0] != "group")  # whole row groups first
    done, size, t0 = 0, 0, time.time()
    with ThreadPoolExecutor(threads) as pool:
        futures = [pool.submit(fetch_unit, how, path, group, rows) for (how, path, group), rows in units]
        for future in as_completed(futures):
            size += future.result()
            done += 1
            if done % 25 == 0 or done == len(units):
                print(f"{time.strftime('%H:%M:%S')} {done}/{len(units)} row groups, {size / 1e9:.2f} GB of images, {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------- comparison with the evaluation images
def dhash(image):
    pixels = list(image.convert("L").resize((17, 16), Image.LANCZOS).getdata())
    return sum(1 << (row * 16 + col) for row in range(16) for col in range(16) if pixels[row * 17 + col] > pixels[row * 17 + col + 1])


def fingerprints(data):
    """SHA-256, and the difference hashes of the whole image and of its central square."""
    image = Image.open(io.BytesIO(data))
    image.draft("RGB", (512, 512))  # JPEG: decode at reduced size, enough for a 17 x 16 hash
    image = image.convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return hashlib.sha256(data).hexdigest(), dhash(image), dhash(image.crop((left, top, left + side, top + side)))


def evaluation_images():
    """(set, file) -> fingerprints of every image of every evaluation set, old and new."""
    sys.path.insert(0, str(Path(__file__).parent))
    from make_test_set import SETS
    paths = []
    for name, spec in SETS.items():
        if Path(spec["listing"]).exists():
            paths += [(name, row["file"], Path(spec["root"]) / row["file"]) for row in csv.DictReader(open(spec["listing"], encoding="utf-8"))]
    with ProcessPoolExecutor(4) as pool:
        prints = list(pool.map(fingerprints, [p.read_bytes() for _, _, p in paths], chunksize=16))
    return {(n, f): fp for (n, f, _), fp in zip(paths, prints)}


def matches(fp, evaluation):
    sha, *hashes = fp
    for key, (esha, *ehashes) in evaluation.items():
        if sha == esha:
            return key, "identical file (SHA-256)"
        if any((a ^ b).bit_count() <= DHASH_LIMIT for a in hashes for b in ehashes):
            return key, "difference hash within 24 bits"
    return None


def finish():
    sel = pq.read_table(META / "selection.parquet").to_pandas()
    raw = pa.concat_tables([pq.read_table(p) for p in sorted(RAW.glob("*.parquet"))]).to_pandas()
    data = sel.merge(raw, on=["file", "group", "index"], how="left", validate="one_to_one")
    missing = data.image_data.isna().sum()
    assert not missing, f"{missing} selected images are not downloaded yet"
    evaluation = evaluation_images()
    print("evaluation images:", len(evaluation), flush=True)
    with ProcessPoolExecutor(4) as pool:
        prints = list(pool.map(fingerprints, data.image_data, chunksize=16))
    data["sha256"] = [fp[0] for fp in prints]
    reasons = []
    for fp, size in zip(prints, data.image_data.map(len)):
        hit = matches(fp, evaluation)
        reasons.append("over 20 MB" if size > MAX_BYTES else f"{hit[1]}: {hit[0][0]}/{hit[0][1]}" if hit else "")
    data["reason"] = reasons
    left = data[data.reason != ""]
    keep = data[data.reason == ""].sample(frac=1, random_state=SEED)  # shuffled, so that every shard mixes all parts
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("*.parquet"):
        old.unlink()
    for i in range(0, len(keep), 500):
        chunk = keep.iloc[i:i + 500]
        pq.write_table(pa.Table.from_pandas(chunk[["image_data", "label", "part", "generator", "sha256"]], preserve_index=False),
                       OUT / f"train-{i // 500:03d}.parquet")
    LISTING.parent.mkdir(parents=True, exist_ok=True)
    columns = ["part", "file", "group", "index", "row_id", "sha256", "label", "generator"]
    keep.assign(label=keep.label.map({1: "ai", 0: "real"}))[columns].rename(columns={"group": "row_group", "index": "row", "label": "class"}) \
        .sort_values(["part", "file", "row_group", "row"]).to_csv(LISTING, index=False)
    left.assign(label=left.label.map({1: "ai", 0: "real"}))[columns + ["reason"]] \
        .rename(columns={"group": "row_group", "index": "row", "label": "class"}).to_csv(LEFT_OUT, index=False)
    print(keep.groupby(["part", "label"]).size().to_string())
    print(f"kept {len(keep)} ({int((keep.label == 0).sum())} photographs, {int(keep.label.sum())} generated), "
          f"{keep.image_data.map(len).sum() / 1e9:.2f} GB; left out {len(left)}: {left.reason.str.split(':').str[0].value_counts().to_dict()}")
    for p in RAW.glob("*.parquet"):  # the downloaded row groups are no longer needed
        p.unlink()


def check():
    """Every training image against every evaluation image, again: the number of matches must be 0."""
    evaluation = evaluation_images()
    shards = sorted(OUT.glob("*.parquet"))
    data = pa.concat_tables([pq.read_table(p, columns=["image_data"]) for p in shards]).column(0).to_pylist()
    with ProcessPoolExecutor(4) as pool:
        prints = list(pool.map(fingerprints, data, chunksize=16))
    listed = {row["sha256"] for row in csv.DictReader(open(LISTING, encoding="utf-8"))}
    sha_hits = sum(fp[0] == e[0] for fp in prints for e in evaluation.values())
    dhash_hits = sum(matches(fp, evaluation) is not None for fp in prints)
    print(f"{len(prints)} training images ({len(listed & {fp[0] for fp in prints})} in the list) against {len(evaluation)} "
          f"evaluation images: {sha_hits} identical files, {dhash_hits} training images within {DHASH_LIMIT} bits of an evaluation image")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["scan", "select", "download", "finish", "check"])
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    META.mkdir(parents=True, exist_ok=True)
    if args.step == "scan":
        for name in CONFIGS:
            meta(name)
    else:
        {"select": select, "download": lambda: download(args.threads), "finish": finish, "check": check}[args.step]()
