"""app.py: the wording of the answer and of the checker's line, through analyse(), and the per-visitor limit of the
public page. Gradio and the detectors are replaced by stand-ins before app.py is imported, so no model is loaded."""
import importlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

import second_opinion

ATEEQQ = "Ateeqq/ai-vs-human-image-detector"
ATEEQQ_NAME = "the published Ateeqq detector"  # SECOND_OPINION_NAME on the AWS server


class StandIn:
    """A detector or checker that gives a fixed answer and remembers the size of the image it was given."""

    def __init__(self, *answer):
        self.answer, self.sizes = answer, []

    def __call__(self, image):
        self.sizes.append(image.size)
        return self.answer


class Failing:
    """A checker that fails, as a published detector that runs out of memory would."""

    def __call__(self, image):
        raise RuntimeError("out of memory")


@pytest.fixture
def load_app(monkeypatch):
    def load(detector, checker=None, model="claude-opus-5-5", hosted=False, name=None):
        gradio = ModuleType("gradio")
        gradio.Image = gradio.Label = gradio.Markdown = lambda *args, **kwargs: None
        gradio.Interface = lambda **kwargs: SimpleNamespace(**kwargs)  # app.demo.description is the text of the page
        gradio.Request = object
        fusion = ModuleType("fusion")
        fusion.classifier = lambda: detector
        monkeypatch.setitem(sys.modules, "gradio", gradio)
        monkeypatch.setitem(sys.modules, "fusion", fusion)
        for variable in ("SPACE_ID", "DETECTOR_PUBLIC", "SECOND_OPINION_NAME"):
            monkeypatch.delenv(variable, raising=False)
        if hosted:
            monkeypatch.setenv("DETECTOR_PUBLIC", "1")
        if name:
            monkeypatch.setenv("SECOND_OPINION_NAME", name)
        if checker:
            monkeypatch.setenv("SECOND_OPINION_MODEL", model)
            monkeypatch.setattr(second_opinion, "classifier", lambda: checker)
        else:
            monkeypatch.delenv("SECOND_OPINION_MODEL", raising=False)
        monkeypatch.delitem(sys.modules, "app", raising=False)
        return importlib.import_module("app")
    return load


def image(width=640, height=480):
    return Image.new("RGB", (width, height))


def request(forwarded=None, host="127.0.0.1"):
    """What Gradio passes as gr.Request behind Caddy: the connection comes from Caddy, the visitor is in the header."""
    return SimpleNamespace(headers={"x-forwarded-for": forwarded} if forwarded else {}, client=SimpleNamespace(host=host))


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


def test_checker_number_is_shown_as_a_separate_line_after_the_detectors_answer(load_app):
    scores, note = load_app(StandIn("ai", 0.9), StandIn(0.97, None), ATEEQQ, name=ATEEQQ_NAME).analyse(image())
    assert scores == pytest.approx({"Real photograph": 0.1, "AI-generated": 0.9})
    assert note == ("**The detector judges this image to be AI-generated** (detector score 90%; this is not a probability)."
                    "\n\nCheck by the published Ateeqq detector: 97% AI-generated")


def test_checker_number_is_shown_even_when_it_contradicts_the_detector(load_app):
    _, note = load_app(StandIn("real", 0.8), StandIn(0.993, None), ATEEQQ, name=ATEEQQ_NAME).analyse(image())
    assert note == ("**The detector judges this image to be a real photograph** (detector score 80%; this is not a "
                    "probability).\n\nCheck by the published Ateeqq detector: 99% AI-generated")


def test_claude_as_checker_gives_a_class_and_no_number(load_app):
    _, note = load_app(StandIn("ai", 0.9), StandIn("real", "Natural sensor noise.")).analyse(image())
    assert note.endswith("\n\nCheck by Claude: a real photograph. Natural sensor noise.")
    assert note.startswith("**The detector judges this image to be AI-generated** (detector score 90%")


CHECKERS = [StandIn(0.97, None), StandIn(0.02, None), StandIn(0.0, None), StandIn("ai", "Waxy skin."),
            StandIn("real", "Natural noise."), StandIn(None, "the daily limit of checks has been reached"), Failing()]


@pytest.mark.parametrize("detector, size", [(("ai", 0.9), (640, 480)), (("real", 0.8), (640, 480)),
                                            (("ai", 0.6), (640, 480)), (("real", 0.95), (3000, 1000))])
@pytest.mark.parametrize("checker", CHECKERS)
def test_answer_is_exactly_the_detector_only_answer_whatever_the_checker_says(load_app, detector, size, checker):
    alone = load_app(StandIn(*detector)).analyse(image(*size))
    scores, note = load_app(StandIn(*detector), checker, ATEEQQ, name=ATEEQQ_NAME).analyse(image(*size))
    assert scores == alone[0]  # same label and confidence
    answer, line = note.rsplit("\n\n", 1)
    assert answer == alone[1]  # same wording
    assert line.startswith(("Check by the published Ateeqq detector: ", "The check by the published Ateeqq detector was unavailable"))
    for word in ("Undetermined", "both judge", "agree", "firm"):
        assert word not in note
    assert note.count("undetermined") == alone[1].count("undetermined")  # only the detector's own warning of a weak decision


def test_checker_failure_still_gives_the_detectors_answer(load_app):
    scores, note = load_app(StandIn("real", 0.9), Failing(), ATEEQQ, name=ATEEQQ_NAME).analyse(image())
    assert scores == pytest.approx({"Real photograph": 0.9, "AI-generated": 0.1})
    assert note == ("**The detector judges this image to be a real photograph** (detector score 90%; this is not a "
                    "probability).\n\nThe check by the published Ateeqq detector was unavailable.")


def test_checker_without_an_answer_says_why(load_app):
    _, note = load_app(StandIn("real", 0.9), StandIn(None, "the daily limit of checks has been reached")).analyse(image())
    assert note.startswith("**The detector judges this image to be a real photograph**")
    assert note.endswith("\n\nThe check by Claude was unavailable: the daily limit of checks has been reached.")


def test_checker_runs_at_the_same_time_as_the_detector(load_app):
    started = threading.Event()

    def checker(image):
        started.set()
        return 0.5, None

    def detector(image):  # returns only once the checker has started
        assert started.wait(10)
        return "ai", 0.9

    assert load_app(detector, checker, ATEEQQ).analyse(image())[1].endswith(": 50% AI-generated")


def test_large_image_detector_sees_the_central_2048_px_checker_the_whole_image(load_app):
    detector, checker = StandIn("ai", 0.95), StandIn(0.9, None)
    _, note = load_app(detector, checker, ATEEQQ, name=ATEEQQ_NAME).analyse(image(3000, 1000))
    assert detector.sizes == [(2048, 1000)] and checker.sizes == [(3000, 1000)]
    assert note.endswith("The image is 3000 x 1000 pixels; the detector analysed only its central 2048 x 1000 pixels."
                         "\n\nCheck by the published Ateeqq detector: 90% AI-generated")


def test_page_says_where_the_image_is_analysed(load_app):
    assert load_app(StandIn("ai", 0.9)).WHERE == "on this computer and is not sent anywhere"
    assert "for the check, by Claude at Anthropic" in load_app(StandIn("ai", 0.9), StandIn("ai", "")).WHERE
    assert "by Claude at Amazon Bedrock (AWS)" in load_app(StandIn("ai", 0.9), StandIn("ai", ""), "anthropic.claude-opus-5-5").WHERE
    assert "Do not upload private images" in load_app(StandIn("ai", 0.9), StandIn(0.5, None), ATEEQQ, hosted=True).WHERE


def test_page_text_presents_the_check_as_separate_from_the_answer(load_app):
    text = load_app(StandIn("ai", 0.9), StandIn(0.5, None), ATEEQQ, name=ATEEQQ_NAME).demo.description
    assert "a check of the same image by the published Ateeqq detector" in text
    assert "The check does not change the detector's answer" in text
    for word in ("undetermined", "agree", "firm", "second opinion"):
        assert word not in text.lower()
    assert "a check of the same image" not in load_app(StandIn("ai", 0.9)).demo.description  # no checker on this computer


def test_public_page_refuses_the_21st_image_in_10_minutes_from_one_address_without_running_the_models(load_app):
    detector, checker = StandIn("ai", 0.9), StandIn(0.95, None)
    app, now = load_app(detector, checker, ATEEQQ, hosted=True), [0]
    app.limit = app.RateLimit(20, 600, clock=lambda: now[0])
    for _ in range(20):
        assert app.analyse(image(), request("203.0.113.7"))[0] is not None
    assert app.analyse(image(), request("203.0.113.7")) == (
        None, "**Not analysed: this page analyses at most 20 images from one address in 10 minutes.** Try again in 10 minutes.")
    assert len(detector.sizes) == len(checker.sizes) == 20
    assert app.analyse(image(), request("198.51.100.2"))[0] is not None  # another visitor is still served
    now[0] = 559
    assert app.analyse(image(), request("203.0.113.7"))[1].endswith("Try again in 1 minute.")
    now[0] = 600
    assert app.analyse(image(), request("203.0.113.7"))[0] is not None


def test_the_address_counted_is_the_last_entry_of_x_forwarded_for_or_else_the_connection(load_app):
    app = load_app(StandIn("real", 0.9), hosted=True)
    assert app.visitor(request("203.0.113.7")) == "203.0.113.7"
    assert app.visitor(request("10.0.0.1, 203.0.113.7")) == "203.0.113.7"  # entries before the last can be set by the client
    assert app.visitor(request(host="192.0.2.5")) == "192.0.2.5"
    for spoofed in range(20):
        app.analyse(image(), request(f"10.0.0.{spoofed}, 203.0.113.7"))
    assert app.analyse(image(), request("10.0.0.99, 203.0.113.7"))[0] is None


def test_no_limit_on_this_computer(load_app):
    detector = StandIn("real", 0.9)
    app = load_app(detector)
    for _ in range(25):
        app.analyse(image(), request("203.0.113.7"))
    assert len(detector.sizes) == 25


def test_rate_limit_slides_and_forgets_visitors_without_a_call_in_the_window(load_app):
    now = [0]
    limit = load_app(StandIn("ai", 0.9)).RateLimit(2, 600, clock=lambda: now[0])
    assert limit.wait("a") == limit.wait("a") == 0
    now[0] = 100
    assert limit.wait("a") == 500 and limit.wait("b") == 0  # a's first call leaves the window at 600
    now[0] = 600
    assert limit.wait("a") == 0
    now[0] = 1300
    assert limit.wait("c") == 0 and list(limit.calls) == ["c"]


def test_rate_limit_counts_exactly_from_many_threads(load_app):
    limit = load_app(StandIn("ai", 0.9)).RateLimit(20, 600)
    with ThreadPoolExecutor(8) as pool:
        waits = list(pool.map(lambda _: limit.wait("203.0.113.7"), range(200)))
    assert waits.count(0) == 20
