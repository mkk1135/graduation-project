"""app.py: what the page shows, through analyse(), with a checker (the public page: a logo, one result, the checker's)
and without (on this computer: the detector's answer), and the per-visitor limit of the public page. Gradio and the
detectors are replaced by stand-ins before app.py is imported, so no model is loaded."""
import base64
import importlib
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace
from xml.etree import ElementTree

import pytest
from PIL import Image

import second_opinion

ROOT = Path(__file__).resolve().parent.parent
ATEEQQ = "Ateeqq/ai-vs-human-image-detector"  # SECOND_OPINION_MODEL and SECOND_OPINION_NAME on the AWS server
ATEEQQ_NAME = "the published Ateeqq detector"
UNAVAILABLE = "The result is unavailable right now. Please try again."
LOCAL_TEXT = ("Upload an image to check whether it is a real photograph or AI-generated. The image is analysed on this "
              "computer and is not sent anywhere.\n\nThe answer is an indication, not proof. On a held-out test of 384 "
              "photographs and generated images of familiar kinds the detector was wrong for 18 images. On images from the "
              "newest generators and from social media it recognised fewer than half of the generated images, so an answer "
              "\"real\" means only that no known trace was found.")


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
    def load(detector, checker=None, model=ATEEQQ, hosted=False, name=ATEEQQ_NAME):
        gradio, drawn = ModuleType("gradio"), []

        class Interface:
            """As gr.Interface (6.32): keeps its arguments and draws the title and the description first."""

            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                start = len(drawn)
                self.render_title_description()
                self.drawn = drawn[start:]  # what was drawn above the input and output components

            def render_title_description(self):
                gradio.Markdown(f"<h1 style='text-align: center; margin-bottom: 1rem'>{self.title}</h1>")
                if self.description:
                    gradio.Markdown(self.description)

        gradio.Image = lambda *args, **kwargs: "Image"
        gradio.Label = lambda *args, **kwargs: "Label"
        gradio.Markdown = lambda value=None, **kwargs: drawn.append(("Markdown", value, kwargs)) or "Markdown"
        gradio.HTML = lambda value=None, **kwargs: drawn.append(("HTML", value, kwargs)) or "HTML"
        gradio.Interface = Interface
        gradio.Request = object
        fusion = ModuleType("fusion")
        fusion.classifier = lambda: detector
        monkeypatch.setitem(sys.modules, "gradio", gradio)
        monkeypatch.setitem(sys.modules, "fusion", fusion)
        for variable in ("SPACE_ID", "DETECTOR_PUBLIC"):
            monkeypatch.delenv(variable, raising=False)
        if hosted:
            monkeypatch.setenv("DETECTOR_PUBLIC", "1")
        monkeypatch.setenv("SECOND_OPINION_NAME", name)  # set as on the server; the page must not show it
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


# Without a checker (on this computer): the detector's answer, as before.

def test_no_image(load_app):
    assert load_app(StandIn("ai", 0.9)).analyse(None) == (None, "Choose an image first.")


def test_detector_alone(load_app):
    scores, note = load_app(StandIn("real", 0.8)).analyse(image())
    assert scores == pytest.approx({"Real photograph": 0.8, "AI-generated": 0.2})
    assert note == "**The detector judges this image to be a real photograph** (detector score 80%; this is not a probability)."


def test_weak_decision_is_undetermined(load_app):
    _, note = load_app(StandIn("ai", 0.6)).analyse(image())
    assert note == ("**The detector judges this image to be AI-generated** (detector score 60%; this is not a probability)."
                    "\n\nThe detector's decision is weak. Treat the image as undetermined.")


def test_large_image_detector_alone_sees_the_central_2048_px(load_app):
    detector = StandIn("ai", 0.95)
    _, note = load_app(detector).analyse(image(3000, 1000))
    assert detector.sizes == [(2048, 1000)]
    assert note == ("**The detector judges this image to be AI-generated** (detector score 95%; this is not a probability)."
                    "\n\nThe image is 3000 x 1000 pixels; the detector analysed only its central 2048 x 1000 pixels.")


def test_page_without_a_checker_keeps_its_text_and_two_outputs(load_app):
    app = load_app(StandIn("ai", 0.9))
    assert app.demo.title == "AI-Generated Image Detection System"
    assert app.demo.description == LOCAL_TEXT
    assert app.demo.outputs == ["Label", "Markdown"]
    assert "Do not upload private images" in load_app(StandIn("ai", 0.9), hosted=True).demo.description


def test_page_without_a_checker_has_no_logo_and_gradios_own_look(load_app):
    for hosted in (False, True):
        app = load_app(StandIn("ai", 0.9), hosted=hosted)
        assert type(app.demo) is sys.modules["gradio"].Interface and app.LOOK == {}  # launch() gets no css or favicon
        assert [kind for kind, _, _ in app.demo.drawn] == ["Markdown", "Markdown"]
        assert app.demo.drawn[1] == ("Markdown", app.demo.description, {})


# With a checker (the public page): a logo, the title, the tagline and only the checker's result, as one line.

def test_page_with_a_checker_has_the_title_one_sentence_and_one_output(load_app):
    app = load_app(StandIn("ai", 0.9), StandIn(0.5, None), hosted=True)
    assert app.demo.title == "AI-Generated Image Detection System"
    assert app.demo.description == "Real or AI? Find out in seconds."
    assert app.demo.outputs == "Markdown"


def test_public_page_draws_the_logo_above_the_title_and_the_tagline_and_no_other_text(load_app):
    app = load_app(StandIn("ai", 0.9), StandIn(0.5, None), hosted=True)
    local = load_app(StandIn("ai", 0.9), hosted=True)
    (logo_kind, logo, logo_options), title, tagline = app.demo.drawn
    assert logo_kind == "HTML" and logo_options == {"elem_id": "logo"}
    assert title == local.demo.drawn[0]  # the title exactly as gr.Interface draws it
    assert tagline == ("Markdown", "<span>Real or AI?</span> <span>Find out in seconds.</span>", {"elem_id": "tagline"})
    text = [re.sub(r"<[^>]*>", "", value).strip() for _, value, _ in app.demo.drawn]  # the words, read as text
    assert [t for t in text if t] == ["AI-Generated Image Detection System", "Real or AI? Find out in seconds."]
    image_tag = re.fullmatch(r'<img src="data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)" alt="([^"]+)" width="112" height="112">', logo)
    assert base64.b64decode(image_tag[1]) == (ROOT / "assets" / "logo.svg").read_bytes()  # embedded, so the CSP allows it
    assert image_tag[2] == "Logo: a magnifying glass over a photo and pixels"


def test_public_page_look_goes_to_launch(load_app):
    look = load_app(StandIn("ai", 0.9), StandIn(0.5, None), hosted=True).LOOK
    assert set(look) == {"css", "favicon_path"} and look["favicon_path"] == str(ROOT / "assets" / "logo.svg")
    css = re.sub(r"\s+", " ", look["css"])
    first, second = "#tagline p span:first-child", "#tagline p span:last-child"
    assert "#logo img" in css and "#tagline p span { display: block; }" in css
    rule = lambda selector: re.search(re.escape(selector) + r" \{([^}]*)\}", css)[1]
    assert "font-size: clamp(1.6rem, 1rem + 2.6vw, 2.6rem)" in rule(first) and "font-weight: 800" in rule(first)
    assert "font-size: clamp(0.96rem, 0.6rem + 1.56vw, 1.56rem)" in rule(second)  # 60% of the first line at any width
    assert "color: #0f172a" in rule(first) and "color: #64748b" in rule(second)  # navy and grey on the light theme
    assert "color: #fbbf24" in rule(".dark " + first) and "color: #94a3b8" in rule(".dark " + second)  # gold and grey
    assert "gradient" not in css and "text-fill" not in css  # no gradient text
    assert "@supports" not in css  # Gradio drops @supports blocks from custom CSS


def test_logo_is_a_text_free_svg_with_a_name():
    svg = ElementTree.parse(ROOT / "assets" / "logo.svg").getroot()
    ns = "{http://www.w3.org/2000/svg}"
    assert svg.tag == f"{ns}svg" and svg.get("viewBox") == "0 0 64 64"
    assert svg.get("role") == "img" and svg.get("aria-label") == svg.find(f"{ns}title").text
    assert not list(svg.iter(f"{ns}text")) and not list(svg.iter(f"{ns}script")) and not list(svg.iter(f"{ns}image"))
    text = (ROOT / "assets" / "logo.svg").read_text(encoding="utf-8")
    assert "href" not in text  # nothing loaded from elsewhere
    assert all(colour in text for colour in ("#0f172a", "#1e3a8a", "#fbbf24", "#f59e0b"))  # navy badge, gold accents


def test_public_page_shows_only_the_checkers_score(load_app):
    detector = StandIn("ai", 0.9)
    assert load_app(detector, StandIn(0.97, None)).analyse(image()) == "**97% AI-generated**"
    assert detector.sizes == [(640, 480)]  # the detector's answer is computed, but not shown


@pytest.mark.parametrize("detector", [("ai", 0.9), ("real", 0.8), ("ai", 0.6), ("real", 0.999)])
@pytest.mark.parametrize("score, shown", [(0.97, "97%"), (0.993, "99%"), (0.5, "50%"), (0.004, "0%"), (0.0, "0%"), (1.0, "100%")])
def test_the_one_result_is_the_checkers_score_whatever_the_detector_says(load_app, detector, score, shown):
    result = load_app(StandIn(*detector), StandIn(score, None), hosted=True).analyse(image(), request("203.0.113.7"))
    assert result == f"**{shown} AI-generated**"  # one line, the number and nothing else
    for word in ("detector", "Ateeqq", "Check", "check", "judges", "undetermined", "weak", "\n"):
        assert word not in result


@pytest.mark.parametrize("answer, shown", [(("ai", "Waxy skin."), "**AI-generated**"), (("real", "Natural noise."), "**Real photograph**")])
def test_claude_as_checker_gives_its_class_without_a_name_or_reason(load_app, answer, shown):
    assert load_app(StandIn("ai", 0.93), StandIn(*answer), "claude-opus-5-5", name="Claude Opus 5.5").analyse(image()) == shown
    assert load_app(StandIn("real", 0.93), StandIn(*answer), "anthropic.claude-opus-5-5").analyse(image()) == shown


@pytest.mark.parametrize("checker", [Failing(), StandIn(None, "the daily limit of checks has been reached"),
                                     StandIn(None, "Claude declined to answer")])
@pytest.mark.parametrize("detector", [("ai", 0.93), ("real", 0.88)])
def test_without_an_answer_from_the_checker_the_detectors_answer_is_not_shown_instead(load_app, checker, detector):
    assert load_app(StandIn(*detector), checker).analyse(image()) == UNAVAILABLE


def test_messages_of_the_public_page_are_its_one_output(load_app):
    app = load_app(StandIn("ai", 0.9), StandIn(0.97, None), hosted=True)
    assert app.analyse(None) == "Choose an image first."


def test_checker_runs_at_the_same_time_as_the_detector(load_app):
    started = threading.Event()

    def checker(image):
        started.set()
        return 0.5, None

    def detector(image):  # returns only once the checker has started
        assert started.wait(10)
        return "ai", 0.9

    assert load_app(detector, checker).analyse(image()) == "**50% AI-generated**"


def test_large_image_detector_sees_the_central_2048_px_checker_the_whole_image(load_app):
    detector, checker = StandIn("ai", 0.95), StandIn(0.9, None)
    assert load_app(detector, checker).analyse(image(3000, 1000)) == "**90% AI-generated**"
    assert detector.sizes == [(2048, 1000)] and checker.sizes == [(3000, 1000)]


def test_public_page_refuses_the_21st_image_in_10_minutes_from_one_address_without_running_the_models(load_app):
    detector, checker = StandIn("ai", 0.9), StandIn(0.95, None)
    app, now = load_app(detector, checker, hosted=True), [0]
    app.limit = app.RateLimit(20, 600, clock=lambda: now[0])
    for _ in range(20):
        assert app.analyse(image(), request("203.0.113.7")) == "**95% AI-generated**"
    assert app.analyse(image(), request("203.0.113.7")) == (
        "**Not analysed: this page analyses at most 20 images from one address in 10 minutes.** Try again in 10 minutes.")
    assert len(detector.sizes) == len(checker.sizes) == 20
    assert app.analyse(image(), request("198.51.100.2")) == "**95% AI-generated**"  # another visitor is still served
    now[0] = 559
    assert app.analyse(image(), request("203.0.113.7")).endswith("Try again in 1 minute.")
    now[0] = 600
    assert app.analyse(image(), request("203.0.113.7")) == "**95% AI-generated**"


# The per-visitor limit.

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
