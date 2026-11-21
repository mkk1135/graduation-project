"""Upload an image and see whether the detector judges it a real photograph or AI-generated.

  python app.py        # then open http://127.0.0.1:7860

The image is analysed on this computer. Nothing is uploaded elsewhere.
The page runs the combined detector (fusion.py): the CNN of this project together with the published
Community Forensics detector. The public demo is a separate page that runs the CNN alone inside the
visitor's browser: space/index.html.
With the environment variable SECOND_OPINION_MODEL set (on the AWS server, see deploy/aws-setup.sh), the page runs
the detector and, at the same time, a checker (second_opinion.py) on the same image, and shows only the checker's
result as one line: the score for "AI-generated" in percent of a published detector run on the same server, or the
class given by Claude Opus 5.5 on the Claude API or on Amazon Bedrock. The detector's answer is computed but not shown,
and it is never shown in place of a missing result. Besides the title, the upload field and the result, the page then
shows only one sentence (TAGLINE).
On a public server the page analyses at most 20 images from one IP address in any 10 minutes.
"""
import logging
import math
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

from fusion import classifier

LARGEST = 2048  # of a bigger image only the central 2048 x 2048 pixels are analysed, at original resolution
classify = classifier()
if os.environ.get("SECOND_OPINION_MODEL"):  # the checker; the variable name is that of the earlier second opinion
    import second_opinion
    checker = second_opinion.classifier()
else:
    checker = None
HOSTED = bool(os.environ.get("SPACE_ID") or os.environ.get("DETECTOR_PUBLIC"))  # DETECTOR_PUBLIC: set by deploy/aws-setup.sh
# The text of the page without a checker (with a checker: TAGLINE only)
WHERE = ("on the server that runs this page, and this page deletes it within two hours. Do not upload private images"
         if HOSTED else "on this computer and is not sent anywhere")
TAGLINE = "Real or AI? Find out in seconds."
RESULT = {"ai": "AI-generated", "real": "Real photograph"}  # the result of a checker that gives a class (Claude)
UNAVAILABLE = "The result is unavailable right now. Please try again."
# Measured on the held-out sets (results/final, results/chat, results/wild; see COMPARISON.md).
MEASURED = ("On a held-out test of 384 photographs and generated images of familiar kinds the detector was wrong for 18 images. "
            "On images from the newest generators and from social media it recognised fewer than half of the generated images, "
            "so an answer \"real\" means only that no known trace was found.")
NAMES = {"ai": "AI-generated", "real": "a real photograph"}
RATE_LIMIT, RATE_WINDOW = 20, 600  # on a public server: at most 20 analyses per visitor in any 10 minutes


class RateLimit:
    """At most `limit` calls per key in any `window` seconds (sliding window), safe to use from several threads.
    Memory: at most `limit` times per key; keys without a call in the last window are dropped once per window."""

    def __init__(self, limit, window, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.calls, self.lock, self.pruned = {}, threading.Lock(), clock()

    def wait(self, key):
        """Counts a call and returns 0; over the limit, counts nothing and returns the seconds until a call is allowed."""
        with self.lock:
            now = self.clock()
            start = now - self.window
            if now - self.pruned >= self.window:
                self.calls = {k: times for k, times in self.calls.items() if times and times[-1] > start}
                self.pruned = now
            times = self.calls.setdefault(key, deque())
            while times and times[0] <= start:
                times.popleft()
            if len(times) >= self.limit:
                return times[0] - start
            times.append(now)
            return 0


limit = RateLimit(RATE_LIMIT, RATE_WINDOW)


def visitor(request):
    """The visitor's IP address. Caddy trusts no proxy, so it replaces an X-Forwarded-For sent by the client with the
    address the connection came from; the last entry is the one Caddy added in any case."""
    forwarded = (getattr(request, "headers", None) or {}).get("x-forwarded-for") or ""
    return forwarded.split(",")[-1].strip() or getattr(getattr(request, "client", None), "host", None)


def shown(text):
    """A message instead of a result: with a checker the page has one output, without it the detector's scores too."""
    return text if checker else (None, text)


def analyse(image, request: gr.Request = None):
    if image is None:
        return shown("Choose an image first.")
    wait = limit.wait(visitor(request)) if HOSTED else 0
    if wait:  # refused before any model runs
        minutes = math.ceil(wait / 60)
        return shown(f"**Not analysed: this page analyses at most {RATE_LIMIT} images from one address in "
                     f"{RATE_WINDOW // 60} minutes.** Try again in {minutes} minute{'s' if minutes > 1 else ''}.")
    image = image.convert("RGB")  # decoded once here: the two threads below must not read the file at the same time
    width, height = image.size
    w, h = min(width, LARGEST), min(height, LARGEST)
    left, top = (width - w) // 2, (height - h) // 2
    with ThreadPoolExecutor(1) as pool:  # the checker runs at the same time as the detector
        checked = pool.submit(checker, image) if checker else None
        kind, confidence = classify(image.crop((left, top, left + w, top + h)))
    if checker:  # the page shows only the checker's result; the detector's answer is not shown
        return result(checked)
    scores = {kind: confidence, "real" if kind == "ai" else "ai": 1 - confidence}
    note = f"**The detector judges this image to be {NAMES[kind]}** (detector score {confidence:.0%}; this is not a probability)."
    if confidence < 0.75:
        note += "\n\nThe detector's decision is weak. Treat the image as undetermined."
    if (w, h) != (width, height):
        note += f"\n\nThe image is {width} x {height} pixels; the detector analysed only its central {w} x {h} pixels."
    return {"Real photograph": scores["real"], "AI-generated": scores["ai"]}, note


def result(checked):
    """The one result shown with a checker: a published detector's score for "AI-generated" in percent, or the class
    given by Claude. Without an answer from the checker the page says so; it never shows the detector's answer instead."""
    try:
        value, why = checked.result()
    except Exception:
        logging.exception("the checker failed")
        return UNAVAILABLE
    if value is None:
        logging.warning("no answer from the checker: %s", why)
        return UNAVAILABLE
    if value in RESULT:  # Claude: a class, no number
        return f"**{RESULT[value]}**"
    return f"**{value:.0%} AI-generated**"


demo = gr.Interface(
    fn=analyse,
    # on a public server the camera is turned off (Permissions-Policy in deploy/aws-setup.sh), so no webcam button
    inputs=gr.Image(type="pil", label="Image (JPEG, PNG or WebP)", sources=["upload", "clipboard"] if HOSTED else None),
    outputs=gr.Markdown() if checker else [gr.Label(num_top_classes=2, label="Result of the detector"), gr.Markdown()],
    title="AI-Generated Image Detection System",
    description=TAGLINE if checker else (
        f"Upload an image to check whether it is a real photograph or AI-generated. The image is analysed {WHERE}.\n\n"
        f"The answer is an indication, not proof. {MEASURED}"
    ),
    flagging_mode="never",
    analytics_enabled=False,
    delete_cache=(3600, 3600),  # every hour, delete the temporary copies of uploads older than an hour
)

if __name__ == "__main__":
    # one image at a time; on a public server at most 20 more may wait, further requests are refused instead of queued
    demo.queue(default_concurrency_limit=1, max_size=20 if HOSTED else None)
    demo.launch(max_file_size="20mb" if HOSTED else None)
