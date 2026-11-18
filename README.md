# AI-Generated Image Detection System

Graduation project (Bachelor of Cybersecurity). It compares four ways of telling real photographs from AI-generated images on one fixed test set, and fine-tunes a CNN that fixes the failure of off-the-shelf detectors on photographs unlike their training data.

Trained model: [mkk1135/ai-image-detector](https://huggingface.co/mkk1135/ai-image-detector)

## Results

48 test images: 26 real photographs and 22 AI-generated images. "Real" is the positive class.

| Model | Accuracy | Precision | Recall | F1 | Real correct | AI correct |
|---|---|---|---|---|---|---|
| Claude Sonnet 5.5, asked "Real or AI-Generated?" (blind) | 97.92% | 96.30% | 100.00% | 98.11% | 26 / 26 | 21 / 22 |
| **CNN fine-tuned in this project, final** (`mkk1135/ai-image-detector`) | 97.92% | 100.00% | 96.15% | 98.04% | 25 / 26 | 22 / 22 |
| ViT detector (`Manish18/deepfake-detection-model`) | 72.92% | 84.21% | 61.54% | 71.11% | 16 / 26 | 19 / 22 |
| CNN fine-tuned in this project, first attempt | 70.83% | 87.50% | 53.85% | 66.67% | 14 / 26 | 20 / 22 |
| SigLIP2 detector (`prithivMLmods/Deep-Fake-Detector-Model`) | 47.92% | 51.72% | 57.69% | 54.55% | 15 / 26 | 8 / 22 |

![Accuracy, precision, recall and F1 of the five models](results/comparison.png)

What the results show:

- The two off-the-shelf detectors are unreliable on these images. Both label a large share of ordinary real photographs as AI-generated.
- The first fine-tuned CNN repeated that failure. It scored 98.4% on held-out images from its own training data and 70.8% here, calling 12 of the 26 real photographs AI-generated.
- The cause was the training data, not the method. The real training images were clean and high-quality only for landscapes and faces, so the model learned to treat clean images of anything else as generated. Adding 7,010 photographs of varied subjects fixed it (see "How the final CNN was chosen").
- The final CNN matches the general-purpose multimodal model on this test set.

### How far to trust these numbers

- With 48 images, one image is worth about 2 percentage points.
- The final CNN's test score is probably optimistic. On the larger check set it scored 88.5% (139 of 157), which is the more cautious estimate of its accuracy on this kind of image.
- Every AI image in the set is unedited generator output, square, and mostly PNG, while the photographs are JPEG. To check that no model is just reading the file format, every image was re-compressed as JPEG at quality 90 and scored again (`results/robustness/`): final CNN 93.75%, ViT 75.00%, first CNN 68.75%, SigLIP2 45.83%.
- The two off-the-shelf detectors are used exactly as published.
- Claude was measured in a single blind pass (see "Blind test of a chat model").

## How the final CNN was chosen

Three training settings were compared on a separate check set of 157 images (69 real, 88 AI-generated) that shares no image and no photographer with the test set. The rule, fixed before the comparison, was that a new setting replaces the previous one only if it is better on the check set both as downloaded and after JPEG re-compression. Only two of the CNNs were ever scored on the test set: the first attempt, and the final model after it had been chosen. The failure of the first attempt on the test set is what prompted the comparison.

| Setting | Check accuracy | Real correct | AI correct | After JPEG 90 |
|---|---|---|---|---|
| 1. Whole image shrunk to 224 px (first attempt) | 79.62% | 41 / 69 | 84 / 88 | 79.62% |
| 2. Crops at original resolution | 76.43% | 33 / 69 | 87 / 88 | 77.07% |
| 3. Setting 2, plus 7,010 photographs and 149 more generators (final) | 88.54% | 58 / 69 | 81 / 88 | 87.90% |

Setting 2 was not adopted. It caught almost every AI image but called more than half of the real photographs generated: it was right on 13 of 15 landscapes and on 1 of 15 food photographs. Blurring a real photograph slightly made it answer "AI" almost every time. That pointed to the training data, which setting 3 corrects.

For comparison, the off-the-shelf detectors score 76.43% (ViT) and 57.32% (SigLIP2) on the check set. All check-set results are in `results/check/`.

## Test set and check set

`results/test_set.csv` and `results/check/check_set.csv` list every image with its author or generator, licence and source URL. The images themselves are not in this repository.

- **Real:** Wikimedia Commons, from the categories Quality images of people, streets, landscapes, buildings, birds and food. In each category the most recently added JPEG files with camera EXIF data and a width of at least 2000 px were taken, one per uploader, at 1920 px width. Drawn on 18 November 2026. Test set: 5, 5, 4, 4, 4 and 4 photographs. Check set: up to 15 per category (69 in total; streets and birds had fewer).
- **AI-generated:** the [Community Forensics evaluation set](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval), from 11 commercial generators: Midjourney v5.2 and v6.1, DALL-E 2 and 3, Firefly Image 2 and 3, Imagen 3, Ideogram v1 and v2, FLUX dev and schnell. Test set: two images each. Check set: eight further images each. None of these generators is in the training data.

```bash
python make_test_set.py
python make_test_set.py --check
```

The Commons draw depends on the day it is run, so the script refuses to overwrite an existing list.

## Training

`train.py` fine-tunes ConvNeXt-Tiny (`facebook/convnext-tiny-224`), a CNN pretrained on ImageNet, for three epochs on about 36,600 images. A random 10% is held out for validation.

| Class | Source | Images |
|---|---|---|
| Generated | [Community Forensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small) (Park and Owens, CVPR 2025): 377 community Stable Diffusion models | about 14,600 |
| Generated | Community Forensics-Small: GLIDE, DeepFloyd IF, BigGAN | 3,000 |
| Real | Community Forensics-Small: COCO (4,800), VISION (1,200), Landscapes HQ (3,000), FFHQ (3,000) | 12,000 |
| Real | [DataSeeds.AI Sample Dataset](https://huggingface.co/datasets/Dataseeds/DataSeeds.AI-Sample-Dataset-DSD): peer-ranked photographs of varied subjects | 7,010 |

Images flagged NSFW by Community Forensics are removed.

The model trains on 256 x 256 crops taken at the images' original resolution, and at inference it sees the whole image without resizing, because shrinking an image removes the fine traces that generators leave. Training images are randomly downscaled, flipped and JPEG-compressed up to twice, so that the model cannot use an image's processing history (generated images are uncompressed PNG, most real ones JPEG) as a shortcut.

```bash
pip install -r requirements.txt
python train.py --hub-id YOUR_HF_USERNAME/ai-image-detector
```

This needs a GPU and downloads about 19 GB; it took 21 minutes of training on an RTX 4070 laptop GPU. `train.ipynb` runs the same script on a free Google Colab GPU, but it has not been tested there. To check the pipeline in two minutes on a CPU, with nothing uploaded:

```bash
python train.py --dry-run
```

## Evaluation

```bash
python evaluate.py MODEL_ID --real data/test/real --ai data/test/ai --name short_name
python evaluate.py MODEL_ID --real data/test/real --ai data/test/ai --name short_name --jpeg 90 --out results/robustness
python evaluate.py --chart
```

Each run writes the metrics, per-image predictions and a confusion matrix.

## Blind test of a chat model

```bash
python blind_test.py prepare
python blind_test.py score answers.txt claude --model "model name"
```

`prepare` writes shuffled copies of the test images with neutral file names and no metadata, all re-encoded as JPEG. The model is shown each copy with the question "Is this image a real photograph taken by a camera, or is it AI-generated / synthetic? Answer with only one word: Real or AI-Generated." and nothing else.

The Claude row above was measured this way on 18 November 2026 with Claude Sonnet 5.5 (`claude-sonnet-5-5`), run as a separate agent that was given only the folder of anonymised copies and told to open nothing else. It was a single pass with no retries; its raw answers are in `results/claude_blind_answers.txt`.
