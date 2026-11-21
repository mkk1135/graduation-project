"""second_opinion.py, the checker of the page: the score for "AI-generated" of a published detector, the image sent to
Claude, and how Claude's answer is read. No request leaves the computer: the Claude client and the published detector
are replaced by stand-ins."""
import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image

import evaluate
import second_opinion
from second_opinion import jpeg

IMAGE = Image.new("RGB", (64, 64))


@pytest.mark.parametrize("size, sent", [((4000, 3000), (1568, 1176)), ((1000, 3000), (523, 1568)), ((800, 600), (800, 600))])
def test_image_is_sent_as_jpeg_with_its_longest_side_at_most_1568_px(size, sent):
    image = Image.open(io.BytesIO(base64.standard_b64decode(jpeg(Image.new("RGBA", size, (10, 200, 30, 128))))))
    assert (image.format, image.mode, image.size) == ("JPEG", "RGB", sent)


def test_jpeg_leaves_the_callers_image_unchanged():
    # app.py hands the same image to the detector at the same time
    image = Image.new("RGB", (4000, 3000))
    jpeg(image)
    assert image.size == (4000, 3000)


@pytest.fixture
def claude(monkeypatch):
    """second_opinion.classifier("claude-...") with a stand-in for the Claude API that gives a fixed reply."""
    def make(reply, stop_reason="end_turn"):
        requests = []

        def create(**request):
            requests.append(request)
            return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=reply)])

        client = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))
        monkeypatch.setattr(second_opinion.anthropic, "Anthropic", lambda **options: client)
        return second_opinion.classifier("claude-opus-5-5"), requests
    return make


def test_claude_answers_are_mapped_to_the_classes_of_the_detector(claude):
    ask, requests = claude(json.dumps({"answer": "AI-Generated", "reason": " Waxy skin and warped text. "}))
    assert ask(IMAGE) == ("ai", "Waxy skin and warped text.")
    content = requests[0]["messages"][0]["content"]
    assert content[0]["source"]["media_type"] == "image/jpeg" and content[1]["text"] == second_opinion.QUESTION
    ask, _ = claude(json.dumps({"answer": "Real", "reason": "Natural sensor noise."}))
    assert ask(IMAGE) == ("real", "Natural sensor noise.")


def test_no_opinion_when_claude_declines_or_the_answer_cannot_be_read(claude):
    ask, _ = claude("", stop_reason="refusal")
    assert ask(IMAGE) == (None, "Claude declined to answer")
    ask, _ = claude("Real, I think")
    assert ask(IMAGE) == (None, "the answer could not be read")


def test_daily_limit_stops_the_paid_requests(claude, monkeypatch):
    monkeypatch.setattr(second_opinion, "DAILY_LIMIT", 2)
    ask, requests = claude(json.dumps({"answer": "Real", "reason": "Sharp detail."}))
    assert [ask(IMAGE)[0] for _ in range(3)] == ["real", "real", None] and len(requests) == 2
    assert ask(IMAGE) == (None, "the daily limit of checks has been reached")


@pytest.mark.parametrize("answer, ai", [(("ai", 0.934), 0.934), (("real", 0.9), 0.1), (("ai", 0.5), 0.5), (("real", 1.0), 0.0)])
def test_a_published_detector_as_checker_gives_its_score_for_ai_generated(monkeypatch, answer, ai):
    # evaluate.load gives the class the detector chose and the score of that class
    loaded = []
    monkeypatch.setattr(evaluate, "load", lambda model: loaded.append(model) or (lambda image: answer))
    result, detail = second_opinion.classifier("Ateeqq/ai-vs-human-image-detector")(IMAGE)
    assert result == pytest.approx(ai) and detail is None and loaded == ["Ateeqq/ai-vs-human-image-detector"]
