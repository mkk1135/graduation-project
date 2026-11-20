"""Upload an image and see whether the detector judges it a real photograph or AI-generated.

  python app.py        # then open http://127.0.0.1:7860

The image is analysed on this computer. Nothing is uploaded elsewhere.
The page runs the combined detector (fusion.py): the CNN of this project together with the published
Community Forensics detector. The public demo is a separate page that runs the CNN alone inside the
visitor's browser: space/index.html.
With the environment variable SECOND_OPINION_MODEL set (on the AWS server, see deploy/aws-setup.sh), Claude on
Amazon Bedrock gives a second opinion (second_opinion.py) and the answer is firm only when both agree.
"""
import os
from concurrent.futures import ThreadPoolExecutor

import gradio as gr

from fusion import classifier

LARGEST = 2048  # of a bigger image only the central 2048 x 2048 pixels are analysed, at original resolution
classify = classifier()
if os.environ.get("SECOND_OPINION_MODEL"):
    import second_opinion
    second, SECOND_NAME = second_opinion.classifier(), os.environ.get("SECOND_OPINION_NAME", "Claude")
else:
    second = None
HOSTED = bool(os.environ.get("SPACE_ID") or os.environ.get("DETECTOR_PUBLIC"))  # DETECTOR_PUBLIC: set by deploy/aws-setup.sh
WHERE = ("on the server that runs this page, and this page deletes it within two hours. Do not upload private images"
         if HOSTED else "on this computer and is not sent anywhere")
if second and os.environ["SECOND_OPINION_MODEL"].startswith("anthropic."):  # Claude: the image leaves the server
    WHERE = (f"on the server that runs this page and, for a second opinion, by {SECOND_NAME} on Amazon Bedrock (AWS). "
             "This page deletes it within two hours. Do not upload private images")
# Measured on the held-out sets (results/final, results/chat, results/wild; see COMPARISON.md).
MEASURED = ("On a held-out test of 384 photographs and generated images of familiar kinds the detector was wrong for 18 images. "
            "On images from the newest generators and from social media it recognised fewer than half of the generated images, "
            "so an answer \"real\" means only that no known trace was found.")
NAMES = {"ai": "AI-generated", "real": "a real photograph"}


def analyse(image):
    if image is None:
        return None, "Choose an image first."
    image = image.convert("RGB")  # decoded once here: the two threads below must not read the file at the same time
    width, height = image.size
    w, h = min(width, LARGEST), min(height, LARGEST)
    left, top = (width - w) // 2, (height - h) // 2
    with ThreadPoolExecutor(1) as pool:  # Claude answers over the network while the detector computes
        asked = pool.submit(second, image) if second else None
        kind, confidence = classify(image.crop((left, top, left + w, top + h)))
        claude, reason = asked.result() if asked else (None, None)
    scores = {kind: confidence, "real" if kind == "ai" else "ai": 1 - confidence}
    detector = f"detector score {confidence:.0%}; this is not a probability"
    verdict = second_opinion.combine(kind, claude) if second else "detector only"
    if verdict == "agree":
        note = f"**The detector and {SECOND_NAME} both judge this image to be {NAMES[kind]}** ({detector})."
    elif verdict == "disagree":
        note = (f"**Undetermined: the detector judges this image to be {NAMES[kind]} ({detector}), "
                f"but {SECOND_NAME} judges it to be {NAMES[claude]}.** Treat the image as undetermined.")
    else:
        note = f"**The detector judges this image to be {NAMES[kind]}** ({detector})."
    if verdict != "disagree" and confidence < 0.75:
        note += "\n\nThe detector's decision is weak. Treat the image as undetermined."
    if claude:
        note += f"\n\nSecond opinion ({SECOND_NAME}): {reason}"
    elif second:
        note += f"\n\nNo second opinion: {reason}."
    if (w, h) != (width, height):
        note += f"\n\nThe image is {width} x {height} pixels; the detector analysed only its central {w} x {h} pixels."
    return {"Real photograph": scores["real"], "AI-generated": scores["ai"]}, note


demo = gr.Interface(
    fn=analyse,
    inputs=gr.Image(type="pil", label="Image (JPEG, PNG or WebP)"),
    outputs=[gr.Label(num_top_classes=2, label="Result of the detector"), gr.Markdown()],
    title="AI-Generated Image Detection System",
    description=(
        f"Upload an image to check whether it is a real photograph or AI-generated. The image is analysed {WHERE}.\n\n"
        + (f"The answer is firm only when the detector and {SECOND_NAME} agree; otherwise the image is reported as undetermined. "
           f"The second opinion comes from {SECOND_NAME}; the figures below are for the detector alone.\n\n" if second else "")
        + f"The answer is an indication, not proof. {MEASURED}"
    ),
    flagging_mode="never",
    analytics_enabled=False,
    delete_cache=(3600, 3600),  # every hour, delete the temporary copies of uploads older than an hour
)

if __name__ == "__main__":
    demo.launch(max_file_size="20mb" if HOSTED else None)
