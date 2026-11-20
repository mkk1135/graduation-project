"""app.py: the wording of the verdict, through analyse(). Gradio and the detectors are replaced by stand-ins before
app.py is imported, so no model is loaded."""
import importlib
import sys
from types import ModuleType

import pytest
from PIL import Image

import second_opinion


class StandIn:
    """A detector or second opinion that gives a fixed answer and remembers the size of the image it was given."""

    def __init__(self, *answer):
        self.answer, self.sizes = answer, []

    def __call__(self, image):
        self.sizes.append(image.size)
        return self.answer


@pytest.fixture
def load_app(monkeypatch):
    def load(detector, second=None, model="claude-opus-5-5"):
        gradio = ModuleType("gradio")
        gradio.Interface = gradio.Image = gradio.Label = gradio.Markdown = lambda *args, **kwargs: None
        fusion = ModuleType("fusion")
        fusion.classifier = lambda: detector
        monkeypatch.setitem(sys.modules, "gradio", gradio)
        monkeypatch.setitem(sys.modules, "fusion", fusion)
        for name in ("SPACE_ID", "DETECTOR_PUBLIC", "SECOND_OPINION_NAME"):
            monkeypatch.delenv(name, raising=False)
        if second:
            monkeypatch.setenv("SECOND_OPINION_MODEL", model)
            monkeypatch.setattr(second_opinion, "classifier", lambda: second)
        else:
            monkeypatch.delenv("SECOND_OPINION_MODEL", raising=False)
        monkeypatch.delitem(sys.modules, "app", raising=False)
        return importlib.import_module("app")
    return load


def image(width=640, height=480):
    return Image.new("RGB", (width, height))


def test_no_image(load_app):
    assert load_app(StandIn("ai", 0.9)).analyse(None) == (None, "Choose an image first.")


def test_detector_alone(load_app):
    scores, note = load_app(StandIn("real", 0.8)).analyse(image())
    assert scores == pytest.approx({"Real photograph": 0.8, "AI-generated": 0.2})
    assert note == "**The detector judges this image to be a real photograph** (detector score 80%; this is not a probability)."


def test_weak_decision_is_undetermined(load_app):
    _, note = load_app(StandIn("ai", 0.6)).analyse(image())
    assert note.startswith("**The detector judges this image to be AI-generated** (detector score 60%")
    assert note.endswith("The detector's decision is weak. Treat the image as undetermined.")


def test_agreement_is_firm(load_app):
    _, note = load_app(StandIn("ai", 0.9), StandIn("ai", "Waxy skin.")).analyse(image())
    assert note == ("**The detector and Claude both judge this image to be AI-generated** (detector score 90%; this is not "
                    "a probability).\n\nSecond opinion (Claude): Waxy skin.")


def test_disagreement_is_undetermined_without_the_weak_warning(load_app):
    _, note = load_app(StandIn("ai", 0.6), StandIn("real", "Natural noise.")).analyse(image())
    assert note.startswith("**Undetermined: the detector judges this image to be AI-generated (detector score 60%; "
                           "this is not a probability), but Claude judges it to be a real photograph.**")
    assert "weak" not in note and note.endswith("Second opinion (Claude): Natural noise.")


def test_missing_second_opinion_falls_back_to_the_detector(load_app):
    _, note = load_app(StandIn("real", 0.9), StandIn(None, "the daily limit of second opinions has been reached")).analyse(image())
    assert note.startswith("**The detector judges this image to be a real photograph**")
    assert note.endswith("No second opinion: the daily limit of second opinions has been reached.")


def test_large_image_detector_sees_the_central_2048_px_second_opinion_the_whole_image(load_app):
    detector, second = StandIn("ai", 0.95), StandIn("ai", "Too smooth.")
    _, note = load_app(detector, second).analyse(image(3000, 1000))
    assert detector.sizes == [(2048, 1000)] and second.sizes == [(3000, 1000)]
    assert note.endswith("The image is 3000 x 1000 pixels; the detector analysed only its central 2048 x 1000 pixels.")


def test_page_says_where_the_image_is_analysed(load_app):
    assert load_app(StandIn("ai", 0.9)).WHERE == "on this computer and is not sent anywhere"
    assert "by Claude at Anthropic" in load_app(StandIn("ai", 0.9), StandIn("ai", "")).WHERE
    assert "by Claude at Amazon Bedrock (AWS)" in load_app(StandIn("ai", 0.9), StandIn("ai", ""), "anthropic.claude-opus-5-5").WHERE
