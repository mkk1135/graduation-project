---
title: AI-Generated Image Detection System
emoji: 🔍
colorFrom: blue
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-sa-4.0
models:
- mkk1135/ai-image-detector
short_description: Check whether an image is a photograph or AI-generated
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# AI-Generated Image Detection System

Choose an image to see whether the detector [mkk1135/ai-image-detector](https://huggingface.co/mkk1135/ai-image-detector) judges it a real photograph or AI-generated. It is the demo of a Bachelor of Cybersecurity graduation project; the code, the lists of test images and all results are at https://github.com/mkk1135/graduation-project.

- **The image stays on your device.** The detector runs inside your browser (ONNX Runtime Web); nothing is uploaded.
- The answer is an indication, not proof. On the project's held-out test set of 384 images the detector classified 82.0% correctly when run as on this page, and 81.5% when it sees whole images, as the Python version does.
- It misses many images from the newest generators and from social media: on images from the generators behind ChatGPT and Gemini it recognised 54 of 96, and on images from Reddit 26 of 60. An answer "real" means only that no known trace was found.
- Of an image larger than 1024 px only the central 1024 x 1024 px are analysed, at original resolution.
- The repository also contains a combined detector, this CNN together with the published Community Forensics detector, which scored 95.3% on the same test set. It runs on your own computer (`python app.py`), not in this page.
- The first visit downloads the detector (about 110 MB). After that an image took 1 to 5 seconds on the laptop used for testing; a phone or an older computer needs longer.

`model.onnx` is produced by `export_onnx.py` in the repository, which also checks that it gives the same output as the PyTorch model.
