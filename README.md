# AI-Generated Image Detection System

Graduation project (Bachelor of Cybersecurity). It fine-tunes a detector that tells real photographs from AI-generated images, and measures it against five published detectors and a general-purpose multimodal model on the same documented images, with a held-out final test set and decision rules that were written down before that set was scored.

- Trained model: [mkk1135/ai-image-detector](https://huggingface.co/mkk1135/ai-image-detector)
- Online demo (runs in your browser, the image stays on your device): [huggingface.co/spaces/mkk1135/ai-image-detector](https://huggingface.co/spaces/mkk1135/ai-image-detector)
- Every number, by set, source and generator: [COMPARISON.md](COMPARISON.md)
- How decisions were taken: [PROTOCOL.md](PROTOCOL.md)

## Results on held-out images

Percent of images classified correctly (real photographs correct / AI images correct). Each approach was scored once, after all decisions had been committed. JPEG 90 means that every image was re-compressed first. Claude Sonnet 5.5 was tested on the images as downloaded only.

<!-- results -->
| Approach | Final test set (384) | Final test set (384), JPEG 90 | ChatGPT and Gemini set (168) | ChatGPT and Gemini set (168), JPEG 90 | Reddit set (120) | Reddit set (120), JPEG 90 |
|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 96.6% (190/181) |  | 85.1% (62/81) |  | 96.7% (59/57) |  |
| Combination: CNN of this project and Community Forensics detector | 95.3% (188/178) | 90.1% (189/157) | 69.0% (71/45) | 62.5% (71/34) | 70.8% (59/26) | 70.8% (60/25) |
| Community Forensics detector, threshold tuned on development data | 94.0% (183/178) | 88.5% (187/153) | 61.3% (70/33) | 58.3% (71/27) | 70.8% (58/27) | 70.0% (57/27) |
| Community Forensics detector | 92.2% (190/164) | 82.3% (190/126) | 57.7% (71/26) | 50.6% (71/14) | 64.2% (60/17) | 65.8% (60/19) |
| CNN of this project, setting 4, threshold tuned on development data | 83.1% (164/155) | 78.4% (173/128) | 69.6% (47/70) | 69.0% (55/61) | 71.7% (56/30) | 70.0% (56/28) |
| CNN of this project, setting 5 before the threshold | 82.8% (165/153) | 80.2% (169/139) | 64.9% (52/57) | 63.1% (57/49) | 68.3% (56/26) | 68.3% (56/26) |
| CNN of this project, setting 4 | 82.3% (147/169) | 81.0% (156/155) | 73.2% (43/80) | 72.6% (50/72) | 79.2% (55/40) | 75.8% (55/36) |
| CNN of this project, setting 5 (published) | 81.5% (171/142) | 78.1% (174/126) | 68.5% (61/54) | 63.1% (63/43) | 68.3% (56/26) | 69.2% (57/26) |
| CNN of this project, first attempt (setting 1) | 77.6% (124/174) | 75.5% (131/159) | 59.5% (41/59) | 58.9% (47/52) | 70.0% (38/46) | 70.8% (39/46) |
| Ateeqq SigLIP detector | 75.8% (160/131) | 75.8% (159/132) | 82.1% (61/77) | 81.0% (58/78) | 83.3% (50/50) | 83.3% (50/50) |
| Manish18 ViT detector | 66.4% (87/168) | 63.3% (87/156) | 70.8% (42/77) | 72.0% (45/76) | 57.5% (22/47) | 57.5% (22/47) |
| umm-maybe Swin detector | 59.4% (165/63) | 56.2% (164/52) | 52.4% (59/29) | 54.2% (63/28) | 59.2% (53/18) | 60.0% (54/18) |
| prithivMLmods SigLIP2 detector | 50.0% (102/90) | 49.7% (101/90) | 57.7% (45/52) | 57.1% (43/53) | 46.7% (26/30) | 46.7% (26/30) |
<!-- /results -->

### What the held-out results show

1. **Images of known kinds (final test set, 384 images).** Claude Sonnet 5.5 96.6%, the combined detector 95.3%, the published Community Forensics detector 92.2%, the CNN of this project 81.5%, the four other published detectors 50.0% to 75.8%. The combination is significantly better than the Community Forensics detector as distributed (exact McNemar test, p = 0.004). Much of that gain comes from the decision threshold: with a threshold tuned on the same development images, that detector alone reaches 94.0%, and what the CNN adds beyond it is not significant (p = 0.302). The difference between the combination and Claude Sonnet 5.5 is not significant either (p = 0.458).
2. **JPEG re-compression.** The Community Forensics detector falls from 92.2% to 82.3%, because it then misses generated images (126 recognised instead of 164); the combination falls from 95.3% to 90.1%.
3. **New generators and images from Reddit.** On images from GPT Image 2, GPT Image 1.5, Nano Banana and Nano Banana Pro the combination scores 69.0% and on images taken from Reddit 70.8%: it accepts nearly every photograph but recognises fewer than half of the generated images. Claude Sonnet 5.5 is the most accurate on both sets (85.1% and 96.7%). The best of the detectors there is the Ateeqq detector (82.1% and 83.3%), which scored only 75.8% on the final test set: no detector was good on every kind of image.
4. **An answer "generated" is more reliable than an answer "real".** When the combination called an image generated, it was right for 98% of those images on the final test set, 98% on the ChatGPT and Gemini set and 96% on the Reddit set. When it called an image real, it was right for 93%, 58% and 63%. On new kinds of image, "real" means only that no known trace was found.
5. **The second round did not improve the CNN itself.** The stronger shrinking (setting 5) and the tuned threshold were adopted by rules fixed in advance. On the final test set the published CNN (81.5%) cannot be told apart from the CNN of the first round, setting 4 (82.3%, p = 0.791). On the other two sets the earlier CNN scores higher: 73.2% against 68.5% on the ChatGPT and Gemini set (p = 0.291, not significant) and 79.2% against 68.3% on the Reddit set (p = 0.002).
6. **A small test set misleads.** An earlier CNN scored 97.9% on the pilot test set of 48 images and 80.7% on the test set of 384 images that followed.

The answer of any of these detectors is an indication, not proof.

## How the evaluation was done

1. **First round.** A pilot test set of 48 images was too small and too much like the data used for tuning: an early model scored far higher on it than on a test set of 384 images (point 6 above). All approaches were then scored once on that larger set (`results/main/`).
2. **Why a second round.** Those results showed where the fine-tuned CNN was weak, so any further change would have been tuned on the test set. The set was demoted to a *development test set*.
3. **Second round.** Three new sets were drawn and held out: a final test set of 384 images, a set of images from the generators behind ChatGPT and Gemini, and a set of images as shared on Reddit. The decision rules (which training setting to adopt, how to choose the threshold, whether to combine detectors, which comparisons count) were committed in [PROTOCOL.md](PROTOCOL.md) before any of these images was scored. `decide.py` applies the rules to the results database. The decisions were committed in commit `7b8e05d`; every approach was then scored once on the three held-out sets, and nothing was changed afterwards.

| Set | Folder | Images | Role |
|---|---|---|---|
| Pilot test set | `results/` | 48 | first round: early scores |
| Check set | `results/check/` | 157 | choosing training settings; processing test |
| Development test set | `results/main/` | 384 | first round: test. Second round: development |
| **Final test set** | `results/final/` | 384 | held out, scored once |
| **ChatGPT and Gemini set** | `results/chat/` | 168 | held out, scored once |
| **Reddit set** | `results/wild/` | 120 | held out, scored once; labels may be noisy |

Each folder holds the list of its images (`*_set.csv`: file, class, source, author or generator, licence, URL), the metrics of every approach, its answer for every image and a confusion matrix. The images themselves are not in this repository; `make_test_set.py` downloads them.

## Try the detector

**In the browser, with nothing to install:** open the [online demo](https://huggingface.co/spaces/mkk1135/ai-image-detector) and choose an image. The demo runs the CNN of this project alone, not the combined detector, inside your browser, so the image never leaves your device. The first visit downloads the detector (about 110 MB). Of an image larger than 1024 px the demo analyses only the central 1024 x 1024 px; run that way, the CNN scores 82.0% on the final test set (`results/final/crop1024/`), against 81.5% for whole images.

**On your own computer:**

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:7860 and upload an image. The page runs the combined detector: the CNN of this project together with the published Community Forensics detector, which is downloaded on first use (87 MB). The image is analysed on your own computer and is not sent anywhere. The page shows the answer, the confidence, and a warning when the decision is weak (confidence below 75%). Of an image wider or taller than 2048 px, both detectors see only the central 2048 x 2048 px.

**From a program:**

```python
from transformers import pipeline

detector = pipeline("image-classification", model="mkk1135/ai-image-detector")
print(detector("photo.jpg"))            # the CNN alone

from PIL import Image
from fusion import classifier           # fusion.py of this repository; reads results/decisions.json

combined = classifier()
print(combined(Image.open("photo.jpg").convert("RGB")))   # ("real" or "ai", confidence)
```

## The detector

`train.py` fine-tunes ConvNeXt-Tiny (`facebook/convnext-tiny-224`), a CNN pretrained on ImageNet, for three epochs on 36,562 images. A random 10% is held out for validation. Images flagged NSFW by Community Forensics are removed.

| Class | Source | Images |
|---|---|---|
| Generated | [Community Forensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small) (Park and Owens, CVPR 2025): 377 community Stable Diffusion models | 14,592 |
| Generated | Community Forensics-Small: GLIDE, DeepFloyd IF, BigGAN | 2,992 |
| Real | Community Forensics-Small: COCO (4,831), VISION (1,153), Landscapes HQ (2,992), FFHQ (2,992) | 11,968 |
| Real | [DataSeeds.AI Sample Dataset](https://huggingface.co/datasets/Dataseeds/DataSeeds.AI-Sample-Dataset-DSD): peer-ranked photographs of varied subjects | 7,010 |

The model trains on 256 x 256 crops taken at the images' original resolution, and at inference it sees the whole image without resizing, because shrinking an image removes the fine traces that generators leave. Training images are randomly shrunk, flipped, blurred, given noise and JPEG-compressed up to twice, so that the model cannot use an image's processing history (generated images are uncompressed PNG, most real ones JPEG) or its sharpness as a shortcut. Five training settings were tried; `train.py` contains settings 4 and 5 (`--setting 4`, `--setting 5`, the default):

| Setting | Change | Effect |
|---|---|---|
| 1 | whole image shrunk to 224 px | first attempt |
| 2 | crops at original resolution | kept fine detail, but took most clean photographs of subjects other than landscapes and faces for generated images |
| 3 | 7,010 photographs of varied subjects and more generated images added | removed that subject-matter shortcut |
| 4 | random blur and noise | the model reads sharpness less as a sign of origin |
| 5 | stronger random shrinking: seven times in ten by any factor down to the crop size, instead of half the time by up to 2x | fewer small web photographs called generated and less sensitivity to blur and resizing, but fewer generated images caught |

After training, the decision threshold of setting 5 (0.69 on the "AI" score) was chosen on the development images and built into the published model. The combined detector (`fusion.py`) takes a weighted mean of the "AI" probabilities of this CNN and of the published Community Forensics detector, with weight 0.25 for the CNN and threshold 0.25; the weight and the threshold were chosen on the development images too.

```bash
python train.py --hub-id YOUR_HF_USERNAME/ai-image-detector    # needs a GPU; downloads about 19 GB
python train.py --seed 1 --out checkpoints/seed1               # a repeated run
python train.py --setting 4 --out checkpoints/s4               # the setting of the first round
python train.py --dry-run                                      # one epoch on 64 images, nothing uploaded
```

## Reproduce the evaluation

```bash
python make_test_set.py final                 # also: test, check, main, chat, wild
python evaluate.py MODEL_ID --real data/final/real --ai data/final/ai --name short_name --out results/final
python evaluate.py MODEL_ID --real data/final/real --ai data/final/ai --name short_name --out results/final/jpeg90 --jpeg 90
python evaluate.py fusion --real data/final/real --ai data/final/ai --name fusion --out results/final
python blind_test.py prepare --set final      # anonymised copies for a chat model
python blind_test.py score answers.txt claude_sonnet_5_5 --model "model name" --set final
python results_db.py                          # database of every prediction, and COMPARISON.md
python decide.py check                        # are the recorded decisions reproduced from the database?
```

The decisions themselves were taken on development data only, with these commands:

```bash
python decide.py setting                      # rule 1: does setting 5 replace setting 4?
python decide.py threshold                    # rule 2; --model commfor or --model cnn_final for the secondary comparisons
python decide.py fusion                       # rule 3: combination with the Community Forensics detector
python apply_threshold.py checkpoints/s5_seed42/checkpoint-3087 cnn_s5_raw models/cnn_s5
python rethreshold.py commfor results/final results/final/jpeg90 results/chat results/chat/jpeg90 results/wild results/wild/jpeg90
```

Set `DETECTOR_DEVICE=cuda` to run the detectors on a GPU.

| Script | Purpose |
|---|---|
| `train.py` | fine-tunes the detector |
| `make_test_set.py` | builds the six evaluation sets and records every image |
| `evaluate.py` | scores a detector; `--jpeg 90` re-compresses first, `--crop 1024` analyses the centre only, `--compare NAME` runs McNemar tests against one model, `--chart` draws a bar chart |
| `commfor.py`, `fusion.py` | the published Community Forensics detector and its combination with the CNN |
| `blind_test.py` | blind test of a chat model |
| `diagnose.py`, `jpeg_quality.py` | how blur, noise, resizing and JPEG strength change a detector's answers |
| `check_overlap.py` | checks that no real evaluation photograph is a training image |
| `analyse_models.py`, `results_db.py` | model files and speed; SQLite database of all results (`results/results.db`) |
| `decide.py`, `apply_threshold.py`, `rethreshold.py` | the decisions of the protocol and their application |
| `app.py`, `space/`, `export_onnx.py` | the two interfaces; `space/check_demo.py` compares the browser demo with the Python detector |

### The results database

`results/results.db` holds every prediction of every approach on every evaluation image: 32,044 predictions on 1,261 images.

| Table or view | Content |
|---|---|
| `images` | one row per evaluation image: set, class, source group, generator or photo category, author, licence, URL |
| `models` | one row per approach: architecture, number of weights, file size and format, input size, licence, speed |
| `predictions` | one row per model, set, condition and image: the answer and its score |
| `processing` | check-set images still classified correctly after blur, sharpening, downscaling or noise |
| `scores`, `by_group`, `by_generator` | views: accuracy per model, set and condition; correct answers per source and per generator |

```bash
sqlite3 results/results.db "select model, accuracy from scores where set_name = 'final' and condition = 'as downloaded' order by 2 desc"
```

A GitHub workflow (`.github/workflows/checks.yml`) runs on every push and pull request. It rebuilds this database with `python results_db.py`, which fails if any result file is not reproduced exactly, and runs `python decide.py check`, which fails if the decisions recorded in `results/decisions.json` are not reproduced from the database.

## Blind test of a chat model

`blind_test.py prepare` writes shuffled copies of the images with neutral file names and no metadata, all re-encoded as JPEG with a longest side of at most 1568 px. The model is shown each copy with the question "Is this image a real photograph taken by a camera, or is it AI-generated / synthetic? Answer with only one word: Real or AI-Generated." and nothing else. Claude Sonnet 5.5 (`claude-sonnet-5-5`) was run as separate agents that were given only a folder of at most 48 anonymised copies and told to open nothing else, in a single pass with no retries. Its raw answers are in each results folder (`claude_blind_answers.txt`).

## Limits

- The final test set was held out, but it comes from the same kinds of source as the development data. The ChatGPT and Gemini set and the Reddit set show how the approaches do on other kinds of image.
- The sets are small. On the final test set the 95% confidence interval of the combination is 92.7 to 97.0%; the ChatGPT and Gemini set has 168 images and the Reddit set 120, so differences of a few points there are not meaningful.
- The images from GPT Image 2 and 1.5 are images that users published, and the two collections of Gemini images are community datasets with short descriptions. In the Reddit set the class of an image is the kind of subreddit it was posted in, so some labels may be wrong.
- **Very large images.** The CNN analyses images at their original resolution, and an image of more than 8 megapixels does not fit into the memory of the 8 GB GPU used. In the evaluation, of such an image the CNN analysed only the central part of at most 2048 x 2048 pixels (`fit` in `evaluate.py`); the published detectors shrink every image themselves and received the whole image, and in the combination only the CNN part was affected. The rule affected no image of the check set, the development test set or the final test set, one photograph of the ChatGPT and Gemini set and many of the Reddit images, most of them photographs (counts in [PROTOCOL.md](PROTOCOL.md)). The local interface is stricter: of any image wider or taller than 2048 px it analyses only the central 2048 x 2048 px, with both detectors. The browser demo analyses only the central 1024 x 1024 px.
- The fine-tuned CNN needs about 3.8 seconds per image on a laptop processor (images of 1.8 megapixels on average).
- Only one multimodal model was tested, once per image, and it saw JPEG copies with a longest side of at most 1568 px. Whether any test image was in the training data of Claude Sonnet 5.5 or of the published detectors is not known.
- The published model is a single training run (seed 42). The two repeated runs (seeds 1 and 2) are trained after the final scoring; until their results are added, the variation between runs is not known.
- Partly synthetic images, edited photographs, screenshots, video and deliberate evasion were not tested.
- The model may be used for non-commercial purposes only (CC BY-NC-SA 4.0, following its training data).
