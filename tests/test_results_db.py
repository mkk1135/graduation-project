"""results_db.py: the helpers that fill the database and write the comparison tables."""
import inspect

import pytest

import evaluate
import results_db
from results_db import auc, image_row, percent, prediction_files, table


def test_percent_is_rounded_once_from_the_counts():
    # 322/384 = 83.854...%; rounding the two-decimal accuracy of the database (83.85) again would give 83.8
    assert percent(322, 384) == "83.9" and percent(313, 384) == "81.5" and percent(1, 3) == "33.3"


@pytest.mark.parametrize("name", ["metrics", "wilson"])
def test_copies_of_the_evaluation_formulas_are_unchanged(name):
    # results_db.py runs with the standard library only, so it keeps its own copy of these functions of evaluate.py
    assert inspect.getsource(getattr(results_db, name)) == inspect.getsource(getattr(evaluate, name))


@pytest.mark.parametrize("real, ai, value", [
    ([0.9, 0.8], [0.1, 0.2], 1.0),          # every real image ranked above every generated one
    ([0.1], [0.9], 0.0),                    # the reverse
    ([0.5, 0.5], [0.5, 0.5], 0.5),          # ties count half
    ([0.9, 0.6, 0.4], [0.5, 0.1], 5 / 6),   # 5 of 6 pairs in the right order
])
def test_auc(real, ai, value):
    assert auc(real, ai) == pytest.approx(value)


@pytest.mark.parametrize("cls, source, name, expected", [
    ("ai", "Community Forensics-Eval", "Dalle3", ("Commercial generators", "Dalle3", "")),
    ("ai", "Community Forensics-Eval", "GALIP", ("Open-source generators", "GALIP", "")),
    ("ai", "OpenFake v2.1, reddit test split", "reddit-ai", ("Reddit, AI subreddits", "Reddit, AI subreddits", "")),
    ("real", "OpenFake v2.1, test split", "docci", ("DOCCI", "DOCCI", "")),
    ("real", "Wikimedia Commons: Quality images of food", "A. Author", ("Wikimedia Commons", "food", "A. Author")),
    ("real", "Unsplash Lite", "A. Author", ("Unsplash", "Unsplash", "A. Author")),
    ("real", "Open Images V7 validation set", "A. Author", ("Open Images", "Open Images", "A. Author")),
])
def test_source_group_of_an_image(cls, source, name, expected):
    row = {"file": f"{cls}/x_1.jpg", "class": cls, "source": source, "author_or_generator": name, "licence": "L", "url": "U"}
    assert image_row("main", row) == ("main", "x_1.jpg", cls, *expected, "L", "U")


def test_prediction_files_and_their_conditions(tmp_path):
    for name in ("cnn_s5_predictions.csv", "cnn_s5_jpeg90_predictions.csv", "cnn_s5.json", "jpeg90/fusion_predictions.csv",
                 "crop1024/cnn_s5_predictions.csv", "training/cnn_s5_predictions.csv"):
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).touch()
    assert [(model, condition) for model, condition, _ in prediction_files(tmp_path)] == [
        ("cnn_s5", "jpeg90"), ("cnn_s5", "as downloaded"), ("cnn_s5", "crop1024"), ("fusion", "jpeg90")]


def test_table_is_markdown_with_empty_cells_for_missing_values():
    assert table(["Approach", "AUC"], [["CNN", 0.9], ["Claude", None]]) == (
        "| Approach | AUC |\n|---|---|\n| CNN | 0.9 |\n| Claude |  |\n")


def test_the_written_comparison_leaves_out_the_detector_of_the_public_page():
    # it stays in the database (MODELS), but not in the models of the written tables, nor in the notes as written
    assert "ateeqq" in results_db.MODELS and "ateeqq" not in results_db.SHOWN
    notes = open("results/comparison_notes.md", encoding="utf-8").read()
    written = results_db.without_hidden(notes)
    assert "Ateeqq" in notes and "Ateeqq" not in written
    assert "6. **Limits of the third round.**" in written and "7. **Limits of the third round.**" not in written
    assert notes.count("\n") - written.count("\n") == 1  # one item of the notes is left out, nothing else
