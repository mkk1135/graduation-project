"""Fine-tune a pretrained CNN to separate real photographs from AI-generated images.

  python train.py --hub-id YOUR_HF_USERNAME/ai-image-detector
  python train.py --dry-run        # one epoch on 64 images, nothing uploaded

A checkpoint is pushed to the Hugging Face Hub after every epoch, so a dropped
Colab session does not lose the run.
"""
import argparse
import io
import os

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torchvision.transforms import v2 as T
from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                          Trainer, TrainingArguments)

# Community Forensics (Park and Owens, CVPR 2025), small version: 278k real photographs
# and 278k images from 4,782 generators, in 186 shards of 2,992 images each.
DATASET = "OwensLab/CommunityForensics-Small"
# Eight shards, about 12 GB: 12,000 generated and 12,000 real images.
SHARDS = [
    52, 8, 65,  # generated: 228 community Stable Diffusion models (latent diffusion)
    83,         # generated: GLIDE and DeepFloyd IF (pixel diffusion), BigGAN
    117,        # real: COCO
    116,        # real: COCO and VISION (full-resolution camera photographs)
    163,        # real: Landscapes HQ
    101,        # real: FFHQ portraits
]
DRY_RUN_SHARDS = [78, 117]  # the two smallest: one generated (35 MB), one real (152 MB)


class Transform:
    """Image bytes -> model inputs. A class, so Windows data-loader workers can pickle it."""

    def __init__(self, tf):
        self.tf = tf

    def __call__(self, batch):
        return {
            "pixel_values": [self.tf(Image.open(io.BytesIO(b)).convert("RGB")) for b in batch["image_data"]],
            "labels": [1 - int(label) for label in batch["label"]],  # dataset: 1 = generated. Ours: 1 = real
        }


def accuracy(prediction):
    return {"accuracy": float((np.argmax(prediction.predictions, axis=1) == prediction.label_ids).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hub-id", help="e.g. username/ai-image-detector")
    parser.add_argument("--base", default="facebook/convnext-tiny-224", help="pretrained CNN to start from")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not (args.hub_id or args.dry_run):
        parser.error("give --hub-id, or --dry-run")

    shards = DRY_RUN_SHARDS if args.dry_run else SHARDS
    data = load_dataset(DATASET, data_files=[f"data/HFCF_small_{n}.parquet" for n in shards], split="train")
    data = data.filter(lambda nsfw: not nsfw, input_columns="nsfw_flag").select_columns(["image_data", "label"])
    if args.dry_run:
        data = data.shuffle(seed=0).select(range(64))
    data = data.train_test_split(test_size=0.1, seed=42)

    processor = AutoImageProcessor.from_pretrained(args.base)
    to_tensor = [T.ToImage(), T.ToDtype(torch.float32, scale=True),
                 T.Normalize(processor.image_mean, processor.image_std)]
    # Generated images in this dataset are PNG and most real ones JPEG. Random JPEG
    # re-compression stops the model from using compression traces as the answer.
    train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.3, 1.0)), T.RandomHorizontalFlip(),
                          T.RandomApply([T.JPEG((30, 95))], p=0.5), *to_tensor])
    eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), *to_tensor])  # same as the base model's processor
    data["train"].set_transform(Transform(train_tf))
    data["test"].set_transform(Transform(eval_tf))

    model = AutoModelForImageClassification.from_pretrained(
        args.base, num_labels=2, id2label={0: "ai", 1: "real"}, label2id={"ai": 0, "real": 1},
        ignore_mismatched_sizes=True)  # replaces the 1000-class ImageNet head

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="checkpoints",
            num_train_epochs=1 if args.dry_run else args.epochs,
            per_device_train_batch_size=8 if args.dry_run else args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=min(8, os.cpu_count()),
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            logging_steps=25,
            remove_unused_columns=False,  # the transform needs the raw image column
            push_to_hub=not args.dry_run,
            hub_model_id=args.hub_id,
            hub_strategy="every_save",
            report_to="none",
        ),
        train_dataset=data["train"],
        eval_dataset=data["test"],
        processing_class=processor,  # saved and pushed with the model
        compute_metrics=accuracy,
    )
    trainer.train()
    print(trainer.evaluate())
    if not args.dry_run:
        trainer.push_to_hub(dataset=DATASET)


if __name__ == "__main__":
    main()
