"""The published Community Forensics detector (Park and Owens, CVPR 2025), used as a baseline.

The model definition and the evaluation transform follow the authors' code
(github.com/JeongsooP/Community-Forensics, MIT licence): a ViT-Small/16 at 384 px with a single
output, where a high value means generated. Weights: OwensLab/commfor-model-384 on the Hugging Face Hub.
"""
import timm
import torch
from huggingface_hub import PyTorchModelHubMixin
from torchvision import transforms as T


class ViTClassifier(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, model_size="small", input_size=384, patch_size=16, **unused):
        super().__init__()
        assert model_size == "small" and patch_size == 16, "only the published ViT-S/16 checkpoints are supported"
        self.vit = timm.create_model(f"vit_small_patch16_{input_size}.augreg_in21k_ft_in1k", pretrained=False, num_classes=1)

    def forward(self, x):
        return self.vit(x)


def classifier(model_id):
    """Returns a function: PIL image -> ("real" or "ai", score)."""
    model = ViTClassifier.from_pretrained(model_id).eval()
    size = model.vit.patch_embed.img_size[0]
    prepare = T.Compose([T.Resize(440 if size == 384 else 256), T.CenterCrop(size), T.ToTensor(),
                         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def classify(image):
        with torch.no_grad():
            generated = torch.sigmoid(model(prepare(image.convert("RGB")).unsqueeze(0)))[0, 0].item()
        return ("ai", generated) if generated > 0.5 else ("real", 1 - generated)

    return classify
