"""Which model files were scored: the weights behind every result file, identified by their SHA-256.

  python provenance.py        # writes results/provenance.json

The result files name a Hub model id or a local folder ("model" in each results/*/NAME.json). This script records,
for every approach, what that name stood for: for a Hub model the revision (commit) in the local Hugging Face
cache from which it was loaded and the SHA-256 of its weight file; for a model of this project the SHA-256 of
model.safetensors in its folder and every cached Hub revision that holds the identical file. Only the standard
library and huggingface_hub are used; no model is loaded. No local path outside the repository is written.
"""
import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import scan_cache_dir

from results_db import ALIASES, SETS, prediction_files

OUT = Path("results/provenance.json")
HELD_OUT = "final"  # the "model" of each approach is taken from the result files of the final test set
WEIGHTS = ("model.safetensors", "pytorch_model.bin")  # what transformers and huggingface_hub load, in this order
COMMFOR = "OwensLab/commfor-model-384"
CNN_HUB = "mkk1135/ai-image-detector"  # fusion.py's default CNN
# The published CNN was uploaded from models/cnn_s5. Used only if that Hub revision is not in the local cache.
UPLOADED = {"models/cnn_s5": {"repo": CNN_HUB, "revision": "37da0ae6bfeea23e5c2c0028116e464eefaebb60", "refs": ["main"],
                              "verified": "uploaded from this file, verified after upload"}}
NOT_LOADED = {("umm-maybe/AI-image-detector", "refs/pr/4"):
              "safetensors conversion fetched by transformers in the background; the scored weights are pytorch_model.bin of main"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def weight_file(folder):
    return next((folder / name for name in WEIGHTS if (folder / name).is_file()), None)


def hub_cache():
    """{repo: {revision: {"refs": [...], "file": name, "sha256": ..., "bytes": ...}}} for every cached model repo."""
    found = {}
    for repo in scan_cache_dir().repos:
        if repo.repo_type != "model":
            continue
        for rev in repo.revisions:
            path = weight_file(rev.snapshot_path)
            if path:
                found.setdefault(repo.repo_id, {})[rev.commit_hash] = {
                    "refs": sorted(rev.refs), "file": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
    return found


def local(folder, cache):
    path = weight_file(Path(folder))
    if path is None:
        return {"folder": folder, "note": "folder not present on the computer that ran the evaluation; its weights cannot be hashed"}
    digest = sha256(path)
    published = [{"repo": repo, "revision": rev, "refs": info["refs"], "verified": "same SHA-256 as the cached Hub file"}
                 for repo, revs in cache.items() for rev, info in revs.items() if info["sha256"] == digest]
    published.sort(key=lambda p: not p["refs"])  # a revision with a branch or tag first
    if not published and folder in UPLOADED:
        published = [UPLOADED[folder]]
    return {"folder": folder, "file": path.name, "sha256": digest, "bytes": path.stat().st_size,
            "published_as": published or "not published"}


def hub(repo, cache, earlier=False):
    revs = cache.get(repo)
    if not revs:
        return {"repo": repo, "note": "not in the local Hugging Face cache"}
    if earlier and repo == CNN_HUB:  # this repository was updated between the rounds
        return {"repo": repo, "note": "the revision of this repository at the time of scoring was not recorded; every cached "
                                      "revision, with the local folder that holds the identical file, is listed under hub_cache"}
    main = [rev for rev, info in revs.items() if "main" in info["refs"]]
    if not main:
        return {"repo": repo, "note": "no cached revision of branch main"}
    info = revs[main[0]]
    record = {"repo": repo, "revision": main[0], "refs": info["refs"], "file": info["file"], "sha256": info["sha256"], "bytes": info["bytes"]}
    other = {rev: {**i, "note": NOT_LOADED.get((repo, i["refs"][0] if i["refs"] else ""), "")} for rev, i in revs.items() if rev != main[0]}
    if other:
        record["other_cached_revisions"] = other
    return record


def resolve(model, cache, earlier=False):
    """What the "model" of a result file stands for. earlier: a name used only on development data."""
    derived = re.fullmatch(r"(\w+), threshold ([\d.]+) on the 'AI' score .*", model)
    if derived:
        return {"derived_from": derived.group(1), "threshold_on_ai_score": float(derived.group(2)),
                "note": f"no model was run: the stored scores of {derived.group(1)} were compared with this threshold (rethreshold.py)"}
    if model.startswith("Claude"):
        return {"note": "chat model reached through its interface (blind_test.py); no weights"}
    if model.startswith("fusion"):
        cnn = model.partition(":")[2] or CNN_HUB
        return {"parts": [resolve(cnn, cache, earlier), resolve(COMMFOR, cache, earlier)]}
    if Path(model).is_dir() or model.startswith(("models/", "checkpoints/")):
        return local(model, cache)
    return hub(model, cache, earlier)


def main():
    cache = hub_cache()
    used = {}  # approach -> {model string: [set or set/condition, ...]}
    for set_name, (_, folder) in SETS.items():
        for name, condition, path in prediction_files(folder):
            approach = ALIASES.get(name, name)
            model = json.loads(path.with_name(path.name.replace("_predictions.csv", ".json")).read_text())["model"]
            where = set_name if condition == "as downloaded" else f"{set_name}/{condition}"
            used.setdefault(approach, {}).setdefault(model, []).append(where)
    approaches = {}
    for approach, models in sorted(used.items()):
        held_out = [m for m, sets in models.items() if HELD_OUT in sets]
        record = {"model_in_result_files": held_out[0] if held_out else None, "used_in": models}
        if not held_out:
            record["note"] = "not scored on the held-out sets"
        record.update(resolve(held_out[0], cache) if held_out else {})
        earlier = {m: resolve(m, cache, earlier=True) for m in models if m not in held_out}
        if earlier:
            record["earlier_sets"] = earlier
        approaches[approach] = record
    extra = {"models/cnn_s4_tuned": local("models/cnn_s4_tuned", cache),
             "models/upload": local("models/upload", cache)}
    extra["models/cnn_s4_tuned"]["note"] = ("setting 4 with its tuned threshold built in (apply_threshold.py); not scored itself, "
                                           "the results of cnn_final_tuned come from the stored scores of cnn_final")
    extra["models/upload"]["note"] = "folder from which an earlier version of the Hub model was uploaded; not scored itself"
    def records(x):  # every nested dict
        if isinstance(x, dict):
            yield x
            x = list(x.values())
        for y in x if isinstance(x, list) else []:
            yield from records(y)

    folders = {r["folder"]: r["sha256"] for r in records([extra, approaches]) if r.get("sha256") and r.get("folder")}
    for revs in cache.values():
        for info in revs.values():
            info["identical_local_folders"] = sorted(f for f, digest in folders.items() if digest == info["sha256"])
    out = {
        "about": "Weights behind every result file. 'model_in_result_files' is the model given to evaluate.py for the held-out "
                 "sets (results/final, results/chat, results/wild); 'used_in' lists every set in which a model name occurs; "
                 "'earlier_sets' resolves names used only on development data. For Hub models the revision is the one of branch "
                 "main in the local Hugging Face cache, which the evaluation used offline. Hashes are SHA-256 of the weight file.",
        "approaches": approaches,
        "other_local_models": extra,
        "hub_cache": {repo: revs for repo, revs in sorted(cache.items())},
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    for approach, record in approaches.items():
        parts = record.get("parts") or [record]
        print(approach, "|", record["model_in_result_files"], "|",
              "; ".join(f"{p.get('folder') or p.get('repo', '')} {p.get('revision', '')[:7]} {p.get('sha256', p.get('note', ''))[:16]}"
                        + (f" = {p['published_as'][0]['repo']}@{p['published_as'][0]['revision'][:7]}" if isinstance(p.get("published_as"), list) else "")
                        for p in parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
