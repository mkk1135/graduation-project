"""Checker of an image, shown on the page of app.py next to the detector's answer: a published detector run on the
same server, which gives its score for "AI-generated", or Claude Opus 5.5 (Claude API or Amazon Bedrock), which gives
a class (real photograph or AI-generated) and a one-sentence reason.

app.py runs it when the environment variable SECOND_OPINION_MODEL is set (deploy/aws-setup.sh sets it on the AWS
server; the variable names are those of the earlier second opinion) and shows its result as a separate line that does
not change the detector's answer. Claude gets the question of the blind test (results/blind_prompt.txt) and the whole
image as a JPEG with its longest side at most 1568 px, as in that test; it is also asked for a one-sentence reason.
Credentials: on the Claude API the key in ANTHROPIC_API_KEY (on the server read by systemd from a file only root can
read); on Bedrock the server's IAM role.
"""
import base64
import io
import json
import os
import time

import anthropic
from anthropic import AnthropicBedrockMantle, BetaRefusalFallbackMiddleware

QUESTION = ("Is this image a real photograph taken by a camera, or is it AI-generated / synthetic? "
            "Judge only from what you see in the image itself. If you are unsure, give your single best answer. "
            "Give the answer and one short sentence on what you based it on.")
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer", "reason"],
          "properties": {"answer": {"type": "string", "enum": ["Real", "AI-Generated"]}, "reason": {"type": "string"}}}
DAILY_LIMIT = int(os.environ.get("SECOND_OPINION_DAILY_LIMIT", "150"))  # the page is public; each call is paid


def jpeg(image, longest=1568):
    image = image.convert("RGB")
    image.thumbnail((longest, longest))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return base64.standard_b64encode(buffer.getvalue()).decode()


def classifier(model=None, region=None):
    """Returns a function: PIL image -> (result, detail).
    For a published detector: (its score for "AI-generated", from 0 to 1; None).
    For Claude: ("ai" or "real", the reason given), or (None, why there is no answer).
    model: a Claude model on the Claude API (claude-opus-5-5; key in ANTHROPIC_API_KEY), a Claude model on Amazon
    Bedrock (anthropic.claude-opus-5-5; the server's IAM role), or one of the published detectors scored by evaluate.py
    (e.g. Ateeqq/ai-vs-human-image-detector), which then runs on this server exactly as it was evaluated."""
    model = model or os.environ["SECOND_OPINION_MODEL"]
    if model.startswith("claude-"):  # refusals are retried by the API itself on a suitable model
        client, extra = anthropic.Anthropic(max_retries=1, timeout=90), {
            "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    elif model.startswith("anthropic."):  # Bedrock has no server-side fallback: the SDK middleware retries refusals
        client, extra = AnthropicBedrockMantle(
            aws_region=region or os.environ.get("AWS_REGION", "us-east-1"), max_retries=1, timeout=90,
            middleware=[BetaRefusalFallbackMiddleware([{"model": "anthropic.claude-opus-4-8"}])]), {}
    else:
        from evaluate import load
        detect = load(model)

        def run(image):  # load() gives the score of the class it chose; the Ateeqq detector has two classes,
            # whose scores (softmax) sum to 1
            kind, score = detect(image)
            return (score if kind == "ai" else 1 - score), None

        return run
    used = {}  # calls per day, counted in this process

    def ask(image):
        today = time.strftime("%Y-%m-%d")
        if used.get(today, 0) >= DAILY_LIMIT:
            return None, "the daily limit of checks has been reached"
        if today not in used:
            used.clear()
        used[today] = used.get(today, 0) + 1
        try:
            response = client.beta.messages.create(
                model=model, max_tokens=4000,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg(image)}},
                    {"type": "text", "text": QUESTION}]}], **extra)
        except anthropic.APIStatusError as error:
            return None, f"the request failed (HTTP {error.status_code})"
        except anthropic.APIConnectionError:
            return None, "Amazon Bedrock could not be reached"
        if response.stop_reason == "refusal":
            return None, "Claude declined to answer"
        text = next((block.text for block in response.content if block.type == "text"), "")
        try:
            answer = json.loads(text)
        except json.JSONDecodeError:
            return None, "the answer could not be read"
        return ("ai" if answer["answer"] == "AI-Generated" else "real"), answer["reason"].strip()

    return ask
