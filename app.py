"""Upload an image and see whether the detector judges it a real photograph or AI-generated.

  python app.py        # then open http://127.0.0.1:7860

The image is analysed on this computer. Nothing is uploaded elsewhere.
The page runs the combined detector (fusion.py): the CNN of this project together with the published
Community Forensics detector. The public demo is a separate page that runs the CNN alone inside the
visitor's browser: space/index.html.
"""
import os

import gradio as gr

from fusion import classifier

LARGEST = 2048  # of a bigger image only the central 2048 x 2048 pixels are analysed, at original resolution
classify = classifier()
WHERE = ("on Hugging Face's servers, and this page deletes it within two hours. Do not upload private images"
         if os.environ.get("SPACE_ID") else "on this computer and is not sent anywhere")
# Measured on the held-out sets (results/final, results/chat, results/wild; see COMPARISON.md).
MEASURED = ("On a held-out test of 384 photographs and generated images of familiar kinds the detector was wrong for 18 images. "
            "On images from the newest generators and from social media it recognised fewer than half of the generated images, "
            "so an answer \"real\" means only that no known trace was found.")


def analyse(image):
    if image is None:
        return None, "Choose an image first."
    width, height = image.size
    w, h = min(width, LARGEST), min(height, LARGEST)
    left, top = (width - w) // 2, (height - h) // 2
    kind, confidence = classify(image.convert("RGB").crop((left, top, left + w, top + h)))
    scores = {kind: confidence, "real" if kind == "ai" else "ai": 1 - confidence}
    answer = "a real photograph" if kind == "real" else "AI-generated"
    note = f"**The detector judges this image to be {answer}** (detector score {confidence:.0%}; this is not a probability)."
    if confidence < 0.75:
        note += "\n\nThis is a weak decision. Treat the image as undetermined."
    if (w, h) != (width, height):
        note += f"\n\nThe image is {width} x {height} pixels; only its central {w} x {h} pixels were analysed."
    return {"Real photograph": scores["real"], "AI-generated": scores["ai"]}, note


demo = gr.Interface(
    fn=analyse,
    inputs=gr.Image(type="pil", label="Image (JPEG, PNG or WebP)"),
    outputs=[gr.Label(num_top_classes=2, label="Result"), gr.Markdown()],
    title="AI-Generated Image Detection System",
    description=(
        f"Upload an image to check whether it is a real photograph or AI-generated. The image is analysed {WHERE}.\n\n"
        f"The answer is an indication, not proof. {MEASURED}"
    ),
    flagging_mode="never",
    analytics_enabled=False,
    delete_cache=(3600, 3600),  # every hour, delete the temporary copies of uploads older than an hour
)

if __name__ == "__main__":
    demo.launch()
