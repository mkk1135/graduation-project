"""Put every result of the project into one small SQLite database and write the comparison from it.

  python results_db.py        # builds results/results.db, checks it against the result files, writes COMPARISON.md

Tables
  images        one row per evaluation image: set, file, class, source group, generator or category, author, licence, URL
  models        one row per approach: what it is and what its files contain (from results/models.json, see analyse_models.py)
  predictions   one row per model, set, condition and image: the answer and its score
  processing    check-set images still classified correctly after blur, sharpening, downscaling or noise (diagnose.py)
Views
  scores        images, real correct, AI correct and accuracy per model, set and condition
  by_group      correct answers per source group (Wikimedia Commons, Unsplash, Open Images, commercial / open-source generators)
  by_generator  correct answers per generator or photo category
Example
  sqlite3 results/results.db "select model, accuracy from scores where set_name = 'main' and condition = 'as downloaded' order by 2 desc"
"""
import csv
import json
import re
import sqlite3
from pathlib import Path

DB = Path("results/results.db")
SETS = {"pilot": ("results/test_set.csv", "results"), "check": ("results/check/check_set.csv", "results/check"),
        "main": ("results/main/main_set.csv", "results/main"), "final": ("results/final/final_set.csv", "results/final"),
        "chat": ("results/chat/chat_set.csv", "results/chat"), "wild": ("results/wild/wild_set.csv", "results/wild")}
ALIASES = {"convnext_resize": "cnn_first", "convnext_crop": "cnn_setting2", "convnext_crop_photos": "cnn_setting3",
           "convnext_robust": "cnn_final"}  # the check-set files are named after the training setting
OPEN_SOURCE = {"Hourglass", "DFGAN", "kandinsky_2_2", "stable_cascade", "GALIP", "LCM_lora_sdxl", "kvikontent_midjourney_v6",
               "LCM_lora_sdv15", "DeciDiffusionV2", "LCM_lora_ssd1b"}
MODELS = {  # name: (label, kind, training data as stated by its publisher)
    "claude_sonnet_5_5": ("Claude Sonnet 5.5", "multimodal model, asked in a blind test", "not a detector; general-purpose model of Anthropic"),
    "commfor": ("Community Forensics detector", "published detector", "the authors' full Community Forensics data: 2.7 million images from 4,803 generators"),
    "commfor_tuned": ("Community Forensics detector, threshold tuned on development data", "published detector, tuned",
                      "as the published detector; only its decision threshold was chosen on the development images of this project"),
    "fusion": ("Combination: CNN of this project and Community Forensics detector", "this project", "weighted mean of the two detectors' scores; weight and threshold chosen on development data"),
    "cnn_s5": ("CNN of this project, setting 5 (published)", "this project", "as setting 4, with stronger shrinking in the augmentation; decision threshold chosen on development data"),
    "cnn_s5_seed1": ("CNN of this project, setting 5, second run", "this project", "as setting 5, another seed"),
    "cnn_s5_seed2": ("CNN of this project, setting 5, third run", "this project", "as setting 5, another seed"),
    "cnn_s5_raw": ("CNN of this project, setting 5 before the threshold", "this project", "as setting 5, with the default threshold of 50%"),
    "cnn_s5_seed1_raw": ("CNN of this project, setting 5, second run, before the threshold", "this project", "as setting 5, another seed, default threshold"),
    "cnn_s5_seed2_raw": ("CNN of this project, setting 5, third run, before the threshold", "this project", "as setting 5, another seed, default threshold"),
    "cnn_final": ("CNN of this project, setting 4", "this project", "36,562 images: 10 shards of Community Forensics-Small and 7,010 DataSeeds photographs; crops at original resolution with blur and noise"),
    "cnn_final_tuned": ("CNN of this project, setting 4, threshold tuned on development data", "this project",
                        "as setting 4; only its decision threshold was chosen on the development images, for comparison with setting 5"),
    "cnn_setting3": ("CNN of this project, setting 3", "this project", "as setting 4, without blur and noise"),
    "cnn_setting2": ("CNN of this project, setting 2", "this project", "23,755 images: 8 shards of Community Forensics-Small; crops at original resolution"),
    "cnn_first": ("CNN of this project, first attempt (setting 1)", "this project", "23,755 images: 8 shards of Community Forensics-Small; whole images shrunk to 224 px"),
    "ateeqq": ("Ateeqq SigLIP detector", "published detector", "card: 60,000 AI-generated and 60,000 human images; card reports 99.2% on its own test split"),
    "umm_maybe": ("umm-maybe Swin detector", "published detector", "card: made in October 2022 for artistic images, 'not a deepfake photo detector'"),
    "vit_baseline": ("Manish18 ViT detector", "published detector", "not stated: the model card is an unfilled template"),
    "siglip2_baseline": ("prithivMLmods SigLIP2 detector", "published detector", "card: its own deepfake data; card reports 94.4% on its own test data"),
}
DIAGNOSED = {"Setting 1": "cnn_first", "Setting 2": "cnn_setting2", "Setting 3": "cnn_setting3", "Setting 4": "cnn_final",
             "Setting 5": "cnn_s5_raw", "Setting 5 with tuned threshold": "cnn_s5", "Setting 4 with tuned threshold": "cnn_final_tuned",
             "OwensLab/commfor-model-384": "commfor", "Ateeqq/ai-vs-human-image-detector": "ateeqq",
             "umm-maybe/AI-image-detector": "umm_maybe", "Manish18/deepfake-detection-model": "vit_baseline",
             "prithivMLmods/Deep-Fake-Detector-Model": "siglip2_baseline"}  # headings in results/check/diagnosis*.txt
SCHEMA = """
CREATE TABLE images (set_name TEXT, file TEXT, class TEXT, source_group TEXT, generator TEXT, author TEXT, licence TEXT, url TEXT,
                     PRIMARY KEY (set_name, file));
CREATE TABLE models (model TEXT PRIMARY KEY, label TEXT, kind TEXT, hub_id TEXT, architecture TEXT, weights_million REAL,
                     weights_mb REAL, safe_weights INTEGER, input TEXT, licence TEXT, created TEXT, downloads_last_month INTEGER,
                     card_words INTEGER, card_unfilled_fields INTEGER, cpu_mean_s REAL, cpu_max_s REAL, training TEXT);
CREATE TABLE predictions (model TEXT, set_name TEXT, condition TEXT, file TEXT, predicted TEXT, score REAL,
                          PRIMARY KEY (model, set_name, condition, file));
CREATE TABLE processing (model TEXT, step TEXT, real_correct INTEGER, real_images INTEGER, ai_correct INTEGER, ai_images INTEGER,
                         PRIMARY KEY (model, step));
CREATE VIEW scores AS
  SELECT p.model, p.set_name, p.condition, COUNT(*) AS images,
         SUM(i.class = 'real' AND p.predicted = 'real') AS real_correct, SUM(i.class = 'real') AS real_images,
         SUM(i.class = 'ai' AND p.predicted = 'ai') AS ai_correct, SUM(i.class = 'ai') AS ai_images,
         ROUND(100.0 * SUM(p.predicted = i.class) / COUNT(*), 2) AS accuracy
  FROM predictions p JOIN images i USING (set_name, file) GROUP BY p.model, p.set_name, p.condition;
CREATE VIEW by_group AS
  SELECT p.model, p.set_name, p.condition, i.source_group, COUNT(*) AS images, SUM(p.predicted = i.class) AS correct
  FROM predictions p JOIN images i USING (set_name, file) GROUP BY p.model, p.set_name, p.condition, i.source_group;
CREATE VIEW by_generator AS
  SELECT p.model, p.set_name, p.condition, i.class, i.generator, COUNT(*) AS images, SUM(p.predicted = i.class) AS correct
  FROM predictions p JOIN images i USING (set_name, file) GROUP BY p.model, p.set_name, p.condition, i.class, i.generator;
"""


def image_row(set_name, row):
    name, author = row["author_or_generator"], ""
    if row["class"] == "ai":
        generator = "Reddit, AI subreddits" if name == "reddit-ai" else name
        group = generator if name == "reddit-ai" else "Open-source generators" if name in OPEN_SOURCE else "Commercial generators"
    elif row["source"].startswith("OpenFake"):
        group = generator = {"docci": "DOCCI", "imagenet": "ImageNet", "reddit-real": "Reddit, photography subreddits"}[name]
    else:
        group = next(s for s in ("Wikimedia Commons", "Unsplash", "Open Images") if s.split()[0] in row["source"])
        generator, author = row["source"].split("Quality images of ")[-1] if group == "Wikimedia Commons" else group, name
    return set_name, Path(row["file"]).name, row["class"], group, generator, author, row["licence"], row["url"]


def prediction_files(folder):
    """(model, condition, path) for every predictions file of one evaluation set."""
    for path in sorted(Path(folder).glob("*_predictions.csv")):
        name = path.name[:-len("_predictions.csv")]
        suffix = re.fullmatch(r"(.+)_(jpeg\d+)", name)
        yield (suffix.group(1), suffix.group(2), path) if suffix else (name, "as downloaded", path)
    for path in sorted(Path(folder).glob("*/*_predictions.csv")):
        if re.fullmatch(r"(jpeg|crop)\d+", path.parent.name):
            yield path.name[:-len("_predictions.csv")], path.parent.name, path


def build():
    DB.unlink(missing_ok=True)
    db = sqlite3.connect(DB)
    db.executescript(SCHEMA)
    facts = json.loads(Path("results/models.json").read_text())
    for model, (label, kind, training) in MODELS.items():
        f = facts.get(model, {})
        size = f.get("image_size")
        shown = "" if not f else "original resolution, not resized" if not f["resizes_input"] else f"resized to {size} x {size} px"
        weights = next((mb for name, mb in f.get("weight_files", {}).items() if name.endswith(("model.safetensors", "pytorch_model.bin"))), None)
        db.execute("INSERT INTO models VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            model, label, kind, f.get("hub_id"), f.get("architecture"), f.get("weights_million"), weights, f.get("safe_weights"), shown,
            f.get("licence"), f.get("created"), f.get("downloads_last_month"), f.get("card_words"), f.get("card_unfilled_fields"),
            f.get("cpu_mean_s"), f.get("cpu_max_s"), training))
    checked = 0
    for set_name, (listing, folder) in SETS.items():
        if not Path(listing).exists():
            continue
        db.executemany("INSERT INTO images VALUES (?,?,?,?,?,?,?,?)",
                       [image_row(set_name, row) for row in csv.DictReader(open(listing, encoding="utf-8"))])
        for name, condition, path in prediction_files(folder):
            model = ALIASES.get(name, name)
            assert model in MODELS, f"unknown model {name} in {path}"
            rows = list(csv.DictReader(open(path)))
            db.executemany("INSERT INTO predictions VALUES (?,?,?,?,?,?)",
                           [(model, set_name, condition, r["file"], r["predicted"], float(r["score"]) if r["score"] else None) for r in rows])
            # Self-check: the database must give the same counts as the result file written by evaluate.py.
            result = json.loads(Path(str(path).replace("_predictions.csv", ".json")).read_text())
            got = db.execute("SELECT real_correct, ai_correct, images FROM scores WHERE model=? AND set_name=? AND condition=?",
                             (model, set_name, condition)).fetchone()
            assert got == (result["tp"], result["tn"], result["images"]), (path, got, result)
            checked += 1
    for path in sorted(Path("results/check").glob("diagnosis*.txt")):
        model = None
        for line in path.read_text(encoding="utf-8").splitlines():
            step = re.match(r"\s+(.+?)\s+real correct\s+(\d+)/(\d+)\s+AI correct\s+(\d+)/(\d+)", line)
            if step:
                db.execute("INSERT INTO processing VALUES (?,?,?,?,?,?)", (model, step.group(1), *map(int, step.groups()[1:])))
            elif line.strip():
                model = DIAGNOSED[line.split(":")[0].strip()]
    # Self-check: the unprocessed row of the processing test must equal the ordinary check-set evaluation.
    for model, real_correct, ai_correct in db.execute("SELECT model, real_correct, ai_correct FROM processing WHERE step = 'as downloaded'").fetchall():
        same = db.execute("SELECT real_correct, ai_correct FROM scores WHERE model=? AND set_name='check' AND condition='as downloaded'", (model,)).fetchone()
        assert same in (None, (real_correct, ai_correct)), (model, same, real_correct, ai_correct)
    db.commit()
    counts = [db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("images", "models", "predictions")]
    print(f"wrote {DB}: {counts[0]} images, {counts[1]} models, {counts[2]} predictions; {checked} result files reproduced exactly")
    return db


def auc(real, ai):
    """Probability that a real image gets a higher 'real' score than a generated one (ties count half)."""
    pairs, rank_sum, i = sorted([(s, 1) for s in real] + [(s, 0) for s in ai]), 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += (i + j + 1) / 2 * sum(label for _, label in pairs[i:j])  # tied scores share the mean rank
        i = j
    return (rank_sum - len(real) * (len(real) + 1) / 2) / (len(real) * len(ai))


assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0 and auc([0.1], [0.9]) == 0.0 and auc([0.5, 0.5], [0.5, 0.5]) == 0.5


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
                      *("| " + " | ".join("" if x is None else str(x) for x in row) + " |" for row in rows)]) + "\n"


def comparison(db):
    q = lambda sql, *args: db.execute(sql, args).fetchall()
    label = dict(q("SELECT model, label FROM models"))
    have = {(m, s, c) for m, s, c in q("SELECT DISTINCT model, set_name, condition FROM predictions")}
    primary = "final" if any(s == "final" for _, s, _ in have) else "main"  # the set of the final measurement
    title = {"pilot": "Pilot test set", "check": "Check set", "main": "Development test set", "final": "Final test set",
             "chat": "ChatGPT and Gemini set", "wild": "Reddit set"}
    size = dict(q("SELECT set_name, COUNT(*) FROM images GROUP BY set_name"))
    ranked = [m for m, in q("SELECT model FROM scores WHERE set_name=? AND condition='as downloaded' ORDER BY accuracy DESC", primary)]
    ranked += [m for m in MODELS if m not in ranked]
    cell = {(m, s, c): f"{acc:.1f}% ({rc}/{ac})" for m, s, c, acc, rc, ac in q("SELECT model, set_name, condition, accuracy, real_correct, ai_correct FROM scores")}
    out = ["# Comparison of the approaches\n",
           "Written by `python results_db.py` from the database `results/results.db`, which holds every prediction of every "
           "approach on every evaluation image. Nothing here is typed by hand, except the notes at the end. The roles of the "
           "sets and the rules by which decisions were taken are in [PROTOCOL.md](PROTOCOL.md).\n",
           "## 1. What each model is\n",
           "From the files of each model (`python analyse_models.py`) and its page on the Hugging Face Hub.\n",
           table(["Approach", "Architecture", "Weights (million)", "File (MB)", "Safe file format", "Input", "Licence", "Published", "Downloads last month", "Model card"],
                 [[lab, arch, wm, mb, {1: "yes (safetensors)", 0: "**no (pickle)**"}.get(safe, ""), inp, lic or ("" if not hub else "none stated"), created, dl,
                   "" if words is None else f"{words} words" + (f", {unfilled} unfilled fields" if unfilled else "")]
                  for lab, arch, wm, mb, safe, inp, lic, created, dl, words, unfilled, hub in q(
                      "SELECT label, architecture, weights_million, weights_mb, safe_weights, input, licence, created, downloads_last_month, "
                      "card_words, card_unfilled_fields, hub_id FROM models WHERE architecture IS NOT NULL ORDER BY kind DESC, weights_million")]),
           "What each publisher says about the training data:\n",
           "\n".join(f"- **{lab}:** {training}." for lab, training in q("SELECT label, training FROM models ORDER BY kind DESC, label")) + "\n",
           "## 2. Accuracy on every set\n",
           "Percent of images classified correctly (real photographs correct / AI images correct). JPEG 90 = every image re-compressed first.\n"]
    for heading, sets in (("Held-out sets (scored once, after all decisions)", ("final", "chat", "wild")), ("Development data", ("pilot", "check", "main"))):
        columns = [(s, c, f"{title[s]} ({size[s]})" + ("" if c == "as downloaded" else ", JPEG 90")) for s in sets if s in size
                   for c in ("as downloaded", "jpeg90") if any((m, s, c) in have for m in MODELS)]
        shown = [m for m in ranked if any((m, s, c) in have for s, c, _ in columns)]
        if columns:
            out += [f"**{heading}**\n", table(["Approach", *(c[2] for c in columns)], [[label[m], *(cell.get((m, s, c), "") for s, c, _ in columns)] for m in shown])]
            readme = Path("README.md")
            text = readme.read_text(encoding="utf-8") if readme.exists() else ""
            if heading.startswith("Held-out") and "<!-- results -->" in text:  # the README shows the held-out results
                start, end = text.index("<!-- results -->"), text.index("<!-- /results -->")
                readme.write_text(text[:start] + "<!-- results -->\n" + out[-1] + text[end:], encoding="utf-8", newline="\n")
    for number, set_name in enumerate([s for s in ("final", "chat", "wild", "main") if any((m, s, "as downloaded") in have for m in MODELS)], 3):
        out.append(f"## {number}. {title[set_name]}: images classified correctly, by source and generator\n")
        shown = [m for m, in q("SELECT model FROM scores WHERE set_name=? AND condition='as downloaded' ORDER BY accuracy DESC", set_name)]
        kinds = q("SELECT class, generator, COUNT(*) FROM images WHERE set_name=? GROUP BY class, generator ORDER BY class DESC, COUNT(*) DESC, generator", set_name)
        got = {(m, g): c for m, g, c in q("SELECT model, generator, correct FROM by_generator WHERE set_name=? AND condition='as downloaded'", set_name)}
        groups = q("SELECT class, source_group, COUNT(*) FROM images WHERE set_name=? GROUP BY class, source_group ORDER BY class DESC, source_group", set_name)
        by = {(m, g): c for m, g, c in q("SELECT model, source_group, correct FROM by_group WHERE set_name=? AND condition='as downloaded'", set_name)}
        if len(groups) < len(kinds):
            out.append(table(["Approach", *(f"{g} ({n})" for _, g, n in groups)], [[label[m], *(by.get((m, g), "") for _, g, _ in groups)] for m in shown]))
        out.append(table(["Images", *(label[m] for m in shown)],
                         [[f"{'Real' if c == 'real' else 'AI'}: {g} ({n})", *(got.get((m, g), "") for m in shown)] for c, g, n in kinds]))
    number = len(out)  # only used to keep the headings below in order
    steps = [s for s, in q("SELECT step FROM processing GROUP BY step ORDER BY MIN(rowid)")]
    if steps:
        out.append("## Sensitivity to image processing (check set)\n")
        out.append("Images of the check set (69 real, 88 AI) still classified correctly after one processing step, as real / AI. "
                   "A detector that reads sharpness or cleanness instead of the way an image was made loses real photographs here.\n")
        cells = {(m, s): (r, a, rn + an) for m, s, r, rn, a, an in q("SELECT model, step, real_correct, real_images, ai_correct, ai_images FROM processing")}
        tested = sorted({m for m, _ in cells}, key=lambda m: -sum(r + a for (mm, _), (r, a, _) in cells.items() if mm == m))
        out.append(table(["Approach", *steps, "Mean accuracy"],
                         [[label[m], *(f"{cells[m, s][0]} / {cells[m, s][1]}" for s in steps),
                           f"{100 * sum(cells[m, s][0] + cells[m, s][1] for s in steps) / sum(cells[m, s][2] for s in steps):.1f}%"] for m in tested]))
    out.append(f"## Ranking quality, confidence and speed ({title[primary].lower()})\n")
    out.append("AUC = probability that a real photograph gets a higher 'real' score than a generated image (1.0 = perfect, 0.5 = guessing); "
               "it does not depend on the decision threshold. Seconds = time per image on this laptop's processor, without a GPU.\n")
    rows = []
    for m in [m for m in ranked if (m, primary, "as downloaded") in have]:
        scored = q("SELECT i.class, CASE WHEN p.predicted='real' THEN p.score ELSE 1 - p.score END, p.predicted = i.class, p.score "
                   "FROM predictions p JOIN images i USING (set_name, file) WHERE p.model=? AND p.set_name=? AND p.condition='as downloaded' AND p.score IS NOT NULL", m, primary)
        if not scored:
            rows.append([label[m], "no scores", "", "", "", ""])
            continue
        strong, weak = [ok for _, _, ok, s in scored if s >= 0.9], [ok for _, _, ok, s in scored if s < 0.75]
        mean_s, max_s = q("SELECT cpu_mean_s, cpu_max_s FROM models WHERE model=?", m)[0]
        rows.append([label[m], f"{auc([p for c, p, _, _ in scored if c == 'real'], [p for c, p, _, _ in scored if c == 'ai']):.3f}",
                     f"{100 * sum(strong) / len(strong):.0f}% of {len(strong)}" if strong else "none",
                     f"{100 * sum(weak) / len(weak):.0f}% of {len(weak)}" if weak else "none", mean_s, max_s])
    out.append(table(["Approach", "AUC", "Correct when the score is 0.9 or more", "Correct when the score is below 0.75", "Seconds per image (mean)", "Seconds (slowest image)"], rows))
    notes = Path("results/comparison_notes.md")
    if notes.exists():
        out.append(notes.read_text(encoding="utf-8"))
    Path("COMPARISON.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"wrote COMPARISON.md (final measurement: {title[primary].lower()})")


if __name__ == "__main__":
    comparison(build())
