"""Fine-tune a pretrained CNN to separate real photographs from AI-generated images.

  python train.py --hub-id YOUR_HF_USERNAME/ai-image-detector
  python train.py                  # same training, model kept in checkpoints/ only
  python train.py --dry-run        # one epoch on 64 images, nothing uploaded
  python train.py --resume         # continue an interrupted run from its last checkpoint
  python train.py --seed 1 --out checkpoints/seed1    # a repeated run, to measure the variation between runs
  python train.py --setting 4 --out checkpoints/s4    # the augmentation of the model published in the first round

Third round (PROTOCOL.md): new data only, made by make_train_set.py and kept on drive D: (ROUND3_DIR):
  python train.py --data round3 --init checkpoints/s5_seed42/checkpoint-3087 --epochs 2 --lr 1e-5 --out D:/ai-detector-round3/checkpoints/A
  python train.py --data round3 --init checkpoints/s5_seed42/checkpoint-3087 --epochs 2 --lr 1e-5 --social --out D:/ai-detector-round3/checkpoints/B
  python train.py --data round3 --out D:/ai-detector-round3/checkpoints/C      # from the base network, 3 epochs, 5e-5

With --hub-id a checkpoint is pushed to the Hugging Face Hub after every epoch,
so a dropped Colab session does not lose the run.
"""
import argparse
import io
import math
import os
import random

import numpy as np
import torch
from datasets import Image as ImageFeature
from datasets import concatenate_datasets, load_dataset
from PIL import Image, ImageFilter
from torchvision.transforms import v2 as T
from transformers import (AutoImageProcessor, AutoModelForImageClassification,
                          Trainer, TrainingArguments, set_seed)

# Community Forensics (Park and Owens, CVPR 2025), small version: 278k real photographs
# and 278k images from 4,782 generators, in 186 shards of 2,992 images each.
DATASET = "OwensLab/CommunityForensics-Small"
# Ten shards, about 15 GB: 17,584 generated and 11,968 real images.
SHARDS = [
    52, 8, 65, 7, 12,  # generated: 377 community Stable Diffusion models (latent diffusion)
    83,                # generated: GLIDE and DeepFloyd IF (pixel diffusion), BigGAN
    117,               # real: COCO
    116,               # real: COCO and VISION (full-resolution camera photographs)
    163,               # real: Landscapes HQ
    101,               # real: FFHQ portraits
]
# 7,010 more real images: peer-ranked photographs of varied subjects (4 GB). The real shards
# above hold clean, high-quality images only of landscapes and faces, and a model trained on
# them alone calls clean photographs of anything else generated.
PHOTOS = "Dataseeds/DataSeeds.AI-Sample-Dataset-DSD"
DRY_RUN_SHARDS = [78, 117]  # the two smallest: one generated (35 MB), one real (152 MB)
CROP = 256  # side of the training crop, in pixels of the original image
ROUND3 = os.path.join(os.environ.get("ROUND3_DIR", "D:/ai-detector-round3"), "train_set")  # parquet files of make_train_set.py


class RandomDownscale:
    """Shrink the image with a random filter before the crop is taken, never below the crop size.

    Setting 5 (the published model): seven times in ten, by any factor down to the crop size, every size being
    equally likely on a log scale. Photographs on the web are mostly far smaller than the camera's image, and
    without strongly shrunk, clean photographs in training the model called small, lightly compressed
    photographs generated.
    Setting 4 (the model of the first round): half the time, by up to 2x.
    A class, so that the choice reaches the data-loader workers.
    """

    def __init__(self, strong):
        self.strong = strong

    def __call__(self, image):
        if random.random() < (0.7 if self.strong else 0.5) and min(image.size) > CROP:
            smallest = CROP / min(image.size)
            scale = math.exp(random.uniform(math.log(smallest), 0)) if self.strong else random.uniform(max(0.5, smallest), 1.0)
            size = (round(image.width * scale), round(image.height * scale))
            image = image.resize(size, random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]))
        return image


class SocialMedia:
    """Candidate B of the third round: half the time, the processing of a sharing platform before the usual augmentation.

    The longer side reduced to 1080, 1280, 1440 or 2048 px (if larger) and the image saved as JPEG (quality 60 to 92,
    4:2:0 chroma subsampling; four times in five) or WebP (quality 60 to 90); three times in ten a second share (quality
    50 to 85); and 15 times in 100 a screenshot: rescaled by 0.75 to 1.5 with a bilinear filter, kept without compression.
    """

    @staticmethod
    def share(image, low, high):
        longest = random.choice([1080, 1280, 1440, 2048])
        if max(image.size) > longest:
            scale = longest / max(image.size)
            size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
            image = image.resize(size, random.choice([Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]))
        buffer = io.BytesIO()
        if random.random() < 0.8:
            image.save(buffer, "JPEG", quality=random.randint(low, high), subsampling=2)  # 2 = 4:2:0
        else:
            image.save(buffer, "WEBP", quality=random.randint(low, min(high, 90)))
        return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")

    def __call__(self, image):
        if random.random() >= 0.5:
            return image
        image = self.share(image, 60, 92)
        if random.random() < 0.3:
            image = self.share(image, 50, 85)
        if random.random() < 0.15:
            scale = random.uniform(0.75, 1.5)
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.BILINEAR)
        return image


def random_blur_noise(image):
    """Sometimes blur the crop or add noise, so that softness or cleanness is not read as a sign of generation."""
    if random.random() < 0.3:
        image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2)))
    if random.random() < 0.3:
        noise = np.random.default_rng(random.getrandbits(32)).normal(0, random.uniform(1, 5), (image.height, image.width, 3))
        image = Image.fromarray(np.clip(np.asarray(image) + noise, 0, 255).astype(np.uint8))
    return image


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
    parser.add_argument("--resume", action="store_true", help="continue from the last checkpoint in the output folder")
    parser.add_argument("--seed", type=int, default=42, help="seed of the training run (the data split is always the same)")
    parser.add_argument("--out", default="checkpoints", help="folder for the checkpoints")
    parser.add_argument("--setting", type=int, choices=[4, 5], default=5, help="augmentation: 5 = published model, 4 = milder shrinking (first round)")
    parser.add_argument("--data", choices=["round2", "round3"], default="round2",
                        help="round2 = Community Forensics shards and DataSeeds photographs (the earlier rounds); round3 = the new data of make_train_set.py")
    parser.add_argument("--init", help="weights to start from (default: --base); e.g. the checkpoint of setting 5")
    parser.add_argument("--social", action="store_true", help="add the processing of a sharing platform (candidate B of the third round)")
    args = parser.parse_args()
    set_seed(args.seed)
    push = bool(args.hub_id) and not args.dry_run  # without --hub-id the model stays in checkpoints/

    if args.data == "round3":
        files = sorted(os.path.join(ROUND3, f) for f in os.listdir(ROUND3) if f.endswith(".parquet"))
        data = load_dataset("parquet", data_files=files, split="train").select_columns(["image_data", "label"])  # 1 = generated
        if args.dry_run:
            data = data.shuffle(seed=0).select(range(64))
    else:
        shards = DRY_RUN_SHARDS if args.dry_run else SHARDS
        data = load_dataset(DATASET, data_files=[f"data/HFCF_small_{n}.parquet" for n in shards], split="train")
        data = data.filter(lambda nsfw: not nsfw, input_columns="nsfw_flag").select_columns(["image_data", "label"])
    if args.dry_run and args.data == "round2":
        data = data.shuffle(seed=0).select(range(64))
    elif args.data == "round2":
        photos = load_dataset(PHOTOS, split="train", columns=["image"]).cast_column("image", ImageFeature(decode=False))
        photos = photos.map(lambda image: {"image_data": image["bytes"], "label": 0},  # 0 = real in this dataset's labels
                            input_columns="image", remove_columns=["image"])
        data = concatenate_datasets([data, photos.cast(data.features)])
    data = data.train_test_split(test_size=0.1, seed=42)

    # No resizing at inference: the model sees the whole image at its original resolution,
    # because shrinking an image to 224 px removes the fine traces that generators leave.
    processor = AutoImageProcessor.from_pretrained(args.base, do_resize=False)
    to_tensor = [T.ToImage(), T.ToDtype(torch.float32, scale=True),
                 T.Normalize(processor.image_mean, processor.image_std)]
    # Training uses crops at original resolution. Generated images in this dataset are
    # uncompressed PNG at their native size, most real ones resized JPEG, so random resizing,
    # blur, noise and JPEG re-compression (none, once or twice) keep the model from using an
    # image's processing history as the answer.
    compress = T.RandomApply([T.JPEG((30, 95))], p=0.5)
    train_tf = T.Compose([*([SocialMedia()] if args.social else []),
                          RandomDownscale(strong=args.setting == 5), T.RandomCrop(CROP, pad_if_needed=True, padding_mode="reflect"),
                          T.RandomHorizontalFlip(), random_blur_noise, compress, compress, *to_tensor])
    eval_tf = T.Compose([T.CenterCrop(CROP), *to_tensor])
    data["train"].set_transform(Transform(train_tf))
    data["test"].set_transform(Transform(eval_tf))

    model = AutoModelForImageClassification.from_pretrained(
        args.init or args.base, num_labels=2, id2label={0: "ai", 1: "real"}, label2id={"ai": 0, "real": 1},
        ignore_mismatched_sizes=True)  # replaces the 1000-class ImageNet head

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            seed=args.seed,
            num_train_epochs=1 if args.dry_run else args.epochs,
            per_device_train_batch_size=8 if args.dry_run else args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=min(4, os.cpu_count()),  # each worker loads the GPU libraries: more runs out of memory
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            logging_steps=25,
            remove_unused_columns=False,  # the transform needs the raw image column
            push_to_hub=push,
            hub_model_id=args.hub_id,
            hub_strategy="every_save",
            report_to="none",
        ),
        train_dataset=data["train"],
        eval_dataset=data["test"],
        processing_class=processor,  # saved and pushed with the model
        compute_metrics=accuracy,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    print(trainer.evaluate())
    if push:
        trainer.push_to_hub(dataset="ComplexDataLab/OpenFake" if args.data == "round3" else DATASET)


if __name__ == "__main__":
    main()
