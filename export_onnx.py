"""Export the published detector to ONNX for the in-browser demo, and check that it gives the same answers.

  python export_onnx.py        # writes space/model.onnx (about 110 MB, not kept in Git)

Needs: pip install onnx onnxscript onnxruntime
"""
from pathlib import Path

import numpy as np
import onnxruntime
import torch
from transformers import AutoModelForImageClassification

MODEL = "mkk1135/ai-image-detector"
OUT = Path("space/model.onnx")


class Logits(torch.nn.Module):
    """pixel_values (1, 3, height, width), normalised as in preprocessor_config.json -> logits (1, 2)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).logits


if __name__ == "__main__":
    detector = AutoModelForImageClassification.from_pretrained(MODEL).eval()
    assert detector.config.id2label == {0: "ai", 1: "real"}, "space/index.html assumes this order"
    model = Logits(detector).eval()
    size = torch.export.Dim("height", min=64), torch.export.Dim("width", min=64)
    torch.onnx.export(model, (torch.rand(1, 3, 320, 256),), OUT, input_names=["pixel_values"], output_names=["logits"],
                      dynamic_shapes={"pixel_values": {2: size[0], 3: size[1]}}, external_data=False)
    session = onnxruntime.InferenceSession(OUT)
    for height, width in ((224, 320), (517, 389), (1024, 1024)):  # any size must work, also ones not divisible by 32
        x = torch.rand(1, 3, height, width) * 4 - 2
        with torch.no_grad():
            expected = model(x).numpy()
        got = session.run(None, {"pixel_values": x.numpy()})[0]
        assert np.abs(got - expected).max() < 1e-3, (height, width, got, expected)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.0f} MB); same logits as the PyTorch model at three sizes")
