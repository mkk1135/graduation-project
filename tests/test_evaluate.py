"""evaluate.py: metrics, 95% interval, McNemar test, label mapping and the rule for very large images."""
import pytest
from PIL import Image

from evaluate import fit, mcnemar, metrics, to_class, wilson


def test_metrics_of_a_known_confusion_matrix():
    # The CNN row of the first report: 4 of 26 real and 21 of 22 generated images correct ("real" is the positive class).
    assert metrics(4, 22, 1, 21) == {"tp": 4, "fn": 22, "fp": 1, "tn": 21,
                                     "accuracy": 52.08, "precision": 80.0, "recall": 15.38, "f1": 25.81}


def test_metrics_when_nothing_is_called_real():
    # No image called real: precision, recall and F1 are 0 instead of a division by zero.
    assert metrics(0, 10, 0, 10) == {"tp": 0, "fn": 10, "fp": 0, "tn": 10,
                                     "accuracy": 50.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_metrics_of_a_perfect_detector():
    result = metrics(30, 0, 0, 20)
    assert [result[k] for k in ("accuracy", "precision", "recall", "f1")] == [100.0] * 4


@pytest.mark.parametrize("correct, n, interval", [
    (50, 100, [40.4, 59.6]),  # textbook value
    (0, 10, [0.0, 27.8]),     # unlike the normal approximation, the interval does not collapse to a point at 0% ...
    (10, 10, [72.2, 100.0]),  # ... or at 100%
])
def test_wilson_interval(correct, n, interval):
    assert wilson(correct, n) == interval


def test_wilson_interval_contains_the_estimate_and_narrows_with_more_images():
    for correct, n in ((1, 7), (322, 384), (366, 384)):
        low, high = wilson(correct, n)
        assert 0 <= low <= 100 * correct / n <= high <= 100
    assert wilson(500, 1000)[1] - wilson(500, 1000)[0] < wilson(50, 100)[1] - wilson(50, 100)[0]


def test_mcnemar_exact_p_values():
    assert mcnemar(0, 0) == 1.0               # no discordant images: no evidence of a difference
    assert mcnemar(10, 0) == 2 / 2 ** 10      # all 10 discordant images favour one model
    assert mcnemar(1, 9) == 2 * (1 + 10) / 2 ** 10
    assert mcnemar(5, 5) == 1.0               # capped at 1 for a perfect tie


def test_mcnemar_is_symmetric():
    assert all(mcnemar(b, c) == mcnemar(c, b) for b in range(8) for c in range(8))


@pytest.mark.parametrize("label, expected", [
    ("real", "real"), ("REAL", "real"), ("Real", "real"), ("human", "real"), (" hum ", "real"),
    ("ai", "ai"), ("FAKE", "ai"), ("Fake", "ai"), ("artificial", "ai"),
])
def test_label_mapping(label, expected):
    assert to_class(label) == expected


def test_unknown_label_stops_instead_of_guessing():
    with pytest.raises(SystemExit, match="'deepfake'"):
        to_class("deepfake")


@pytest.mark.parametrize("size", [(4000, 1000), (4000, 2000), (224, 224)])  # 4, exactly 8 and 0.05 megapixels
def test_fit_keeps_images_up_to_8_megapixels(size):
    image = Image.new("RGB", size)
    assert fit(image) is image


@pytest.mark.parametrize("size, kept", [((6000, 1500), (2048, 1500)), ((3000, 3000), (2048, 2048)), ((1000, 9000), (1000, 2048))])
def test_fit_crops_larger_images_to_at_most_2048_pixels(size, kept):
    assert fit(Image.new("RGB", size)).size == kept


def test_fit_keeps_the_central_part():
    image = Image.new("L", (6000, 1500))
    left = (6000 - 2048) // 2
    image.putpixel((left, 0), 255)              # first column of the central part
    image.putpixel((left + 2047, 1499), 128)    # last column
    image.putpixel((left - 1, 0), 7)            # just outside
    cropped = fit(image)
    assert cropped.getpixel((0, 0)) == 255 and cropped.getpixel((2047, 1499)) == 128 and cropped.histogram()[7] == 0
