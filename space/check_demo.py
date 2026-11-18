"""Check the in-browser demo against the PyTorch detector: same images, same answers?

  python space/check_demo.py URL IMAGE [IMAGE ...]      (run from the project folder)

URL is the demo page, for example https://mkk1135-ai-image-detector.static.hf.space/
Each image is uploaded to the page in a real browser, and the probability the page reports is
compared with the PyTorch detector on the same central 1024 x 1024 pixels.
Needs: pip install playwright, and Microsoft Edge.
"""
import sys

from playwright.sync_api import sync_playwright
from transformers import pipeline
from transformers.image_utils import load_image

LARGEST = 1024  # as in index.html
STATUS = "document.getElementById('status').textContent"


def reference(detector, path):
    image = load_image(path)
    w, h = min(image.width, LARGEST), min(image.height, LARGEST)
    left, top = (image.width - w) // 2, (image.height - h) // 2
    scores = {r["label"]: r["score"] for r in detector(image.crop((left, top, left + w, top + h)))}
    return scores["real"], f"{image.width}x{image.height}"


if __name__ == "__main__":
    url, images = sys.argv[1], sys.argv[2:]
    detector = pipeline("image-classification", model="mkk1135/ai-image-detector")
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_function(f"/^Ready|could not/.test({STATUS})", timeout=600000)
        print(f"{url} | {page.inner_text('#status')} | threads: {page.evaluate('ort.env.wasm.numThreads')}")
        worst, slowest = 0, 0
        for path in images:
            page.evaluate(f"{STATUS} = ''")
            page.set_input_files("#file", path)
            page.wait_for_function(f"/^(Done|This image)/.test({STATUS})", timeout=600000)
            status = page.inner_text("#status")
            assert status.startswith("Done"), (path, status)
            got, seconds = float(page.get_attribute("#result", "data-real")), float(status.split()[2])
            want, size = reference(detector, path)
            worst, slowest = max(worst, abs(got - want)), max(slowest, seconds)
            print(f"{path.split('/')[-1]:26s} {size:10s} browser real={got:.4f}  PyTorch real={want:.4f}  {seconds:4.1f} s")
        browser.close()
    print(f"{len(images)} images: largest difference between browser and PyTorch {worst:.4f}, slowest image {slowest:.1f} s")
    assert worst < 0.005, "the page and the PyTorch detector disagree"
