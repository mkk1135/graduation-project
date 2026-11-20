"""Second opinion on an image from Claude Opus 5.5 on Amazon Bedrock: real photograph or AI-generated?

app.py asks it when the environment variable SECOND_OPINION_MODEL is set (deploy/aws-setup.sh sets it on the AWS
server) and shows a firm answer only when the detector and Claude agree. Claude gets the question of the blind test
(results/blind_prompt.txt) and the whole image as a JPEG with its longest side at most 1568 px, as in that test; it is
also asked for a one-sentence reason. Credentials come from the server's IAM role: no key is stored on the server.
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
    """Returns a function: PIL image -> ("ai" or "real", reason), or (None, why there is no answer)."""
    model = model or os.environ["SECOND_OPINION_MODEL"]  # e.g. anthropic.claude-opus-5-5
    client = AnthropicBedrockMantle(aws_region=region or os.environ.get("AWS_REGION", "us-east-1"), max_retries=1, timeout=90,
                                    middleware=[BetaRefusalFallbackMiddleware([{"model": "anthropic.claude-opus-4-8"}])])
    used = {}  # calls per day, counted in this process

    def ask(image):
        today = time.strftime("%Y-%m-%d")
        if used.get(today, 0) >= DAILY_LIMIT:
            return None, "the daily limit of second opinions has been reached"
        if today not in used:
            used.clear()
        used[today] = used.get(today, 0) + 1
        try:
            response = client.beta.messages.create(
                model=model, max_tokens=4000,
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": jpeg(image)}},
                    {"type": "text", "text": QUESTION}]}])
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


def combine(detector, claude):
    """The page's verdict: firm only when both say the same. detector, claude: "ai", "real" or None (no answer)."""
    if claude is None:
        return "detector only"
    return "agree" if detector == claude else "disagree"


assert combine("ai", "ai") == combine("real", "real") == "agree"
assert combine("ai", "real") == combine("real", "ai") == "disagree"
assert combine("real", None) == "detector only"
