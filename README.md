<div align="center">

# AI-Generated Image Detection System

*Graduation project (Bachelor of Cybersecurity).*

[![checks](https://img.shields.io/github/actions/workflow/status/mkk1135/graduation-project/checks.yml?branch=main&label=checks)](https://github.com/mkk1135/graduation-project/actions/workflows/checks.yml) [![Python 3.13.2](https://img.shields.io/badge/python-3.13.2-3776AB?logo=python&logoColor=white)](requirements.lock.txt) [![Model licence: CC BY-NC-SA 4.0](https://img.shields.io/badge/model%20licence-CC%20BY--NC--SA%204.0-lightgrey)](#limits) [![Trained model: mkk1135/ai-image-detector](https://img.shields.io/badge/trained%20model-mkk1135%2Fai--image--detector-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/mkk1135/ai-image-detector) [![Online demo: runs in your browser](https://img.shields.io/badge/online%20demo-runs%20in%20your%20browser-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/spaces/mkk1135/ai-image-detector) [![Public page: until 29 November 2026](https://img.shields.io/badge/public%20page-until%2029%20November%202026-FF9900)](https://35-169-35-59.sslip.io)

[Results on held-out images](#results-on-held-out-images) · [How the evaluation was done](#how-the-evaluation-was-done) · [Try the detector](#try-the-detector) · [The detector](#the-detector) · [Reproduce the evaluation](#reproduce-the-evaluation) · [Blind test of a chat model](#blind-test-of-a-chat-model) · [Limits](#limits)

</div>

---

It fine-tunes a detector that tells real photographs from AI-generated images, and measures it against five published detectors and a general-purpose multimodal model on the same documented images, with a held-out final test set and decision rules that were written down before that set was scored.

- **Trained model:** [mkk1135/ai-image-detector](https://huggingface.co/mkk1135/ai-image-detector)
- **Online demo** (runs in your browser, the image stays on your device): [huggingface.co/spaces/mkk1135/ai-image-detector](https://huggingface.co/spaces/mkk1135/ai-image-detector)
- **Public page, until 29 November 2026** (it shows the score of a published detector from the Hugging Face Hub; the image is analysed on the server and deleted within two hours): [35-169-35-59.sslip.io](https://35-169-35-59.sslip.io)
- **Every number, by set, source and generator:** [COMPARISON.md](COMPARISON.md)
- **How decisions were taken:** [PROTOCOL.md](PROTOCOL.md)

<!-- live-system: the one sentence that names the system of the public demonstration; change it when that changes -->
> [!IMPORTANT]
> **The public demonstration keeps the system of the second round.** By the student's choice, the online demo and the model on Hugging Face and the public page on Amazon Web Services deliberately keep the system of the second round (the public page runs it but shows the score of a published detector as its result; see below); the system of the third round is evaluated here but not deployed. The third-round system is better on new kinds of image, but it flags more photographs as AI-generated and is slightly less accurate on final test set 3.

## Results on held-out images

Percent of images classified correctly (real photographs correct / AI images correct). In each round every approach was scored once on that round's held-out sets, after all decisions of the round had been committed. JPEG 90 means that every image was re-compressed first. Claude Sonnet 5.5 was tested in the second round only, on the images as downloaded. The Seedream 5.0 control holds generated images only.

<!-- results -->
**Third round: held-out sets (scored once, after the decisions of the third round)**

| Approach | Final test set 3 (384) | Final test set 3 (384), JPEG 90 | ChatGPT and Gemini set 3 (384) | ChatGPT and Gemini set 3 (384), JPEG 90 | Reddit set 3 (384) | Reddit set 3 (384), JPEG 90 | Seedream 5.0 control (48) | Seedream 5.0 control (48), JPEG 90 |
|---|---|---|---|---|---|---|---|---|
| Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | 94.5% (189/174) | 89.8% (189/156) | 75.5% (191/99) | 69.8% (191/77) | 71.9% (189/87) | 70.8% (189/83) | 43.8% (0/21) | 45.8% (0/22) |
| Combination of the third round, with the run of candidate B with seed 1 | 93.8% (181/179) | 90.4% (181/166) | 79.7% (170/136) | 78.1% (177/123) | 82.6% (183/134) | 82.3% (183/133) | 81.2% (0/39) | 83.3% (0/40) |
| Combination of the third round, with the run of candidate B with seed 2 | 93.2% (181/177) | 91.4% (183/168) | 80.5% (175/134) | 78.9% (181/122) | 82.0% (184/131) | 81.5% (184/129) | 75.0% (0/36) | 81.2% (0/39) |
| Combination of the third round: CNN of this project (candidate B) and Community Forensics detector | 92.4% (176/179) | 90.6% (179/169) | 81.8% (170/144) | 80.2% (175/133) | 84.1% (181/142) | 84.1% (182/141) | 85.4% (0/41) | 87.5% (0/42) |
| Community Forensics detector, threshold tuned on the development images of the third round | 91.9% (177/176) | 87.2% (179/156) | 73.7% (189/94) | 70.6% (189/82) | 69.0% (174/91) | 69.8% (175/93) | 50.0% (0/24) | 52.1% (0/25) |
| Community Forensics detector | 88.8% (190/151) | 83.1% (190/129) | 63.8% (192/53) | 61.5% (192/44) | 62.0% (190/48) | 61.7% (190/47) | 31.2% (0/15) | 25.0% (0/12) |
| CNN of this project, setting 5 (published) | 82.8% (172/146) | 81.2% (177/135) | 71.9% (155/121) | 69.8% (169/99) | 73.2% (184/97) | 72.9% (185/95) | 39.6% (0/19) | 37.5% (0/18) |
| CNN of this project, third round (adopted candidate) | 82.6% (178/139) | 78.4% (178/123) | 77.6% (164/134) | 77.9% (170/129) | 83.1% (183/136) | 82.3% (183/133) | 79.2% (0/38) | 79.2% (0/38) |
| CNN of this project, third round, candidate A | 82.0% (176/139) | 80.5% (176/133) | 76.6% (156/138) | 77.3% (165/132) | 82.8% (183/135) | 81.8% (183/131) | 77.1% (0/37) | 77.1% (0/37) |
| CNN of this project, setting 4 (first round) | 81.8% (142/172) | 81.0% (154/157) | 67.4% (109/150) | 67.4% (124/135) | 83.1% (177/142) | 80.7% (178/132) | 47.9% (0/23) | 45.8% (0/22) |
| CNN of this project, third round, candidate B, run with seed 1 | 81.5% (173/140) | 78.9% (176/127) | 76.3% (153/140) | 76.0% (159/133) | 84.1% (182/141) | 82.8% (180/138) | 81.2% (0/39) | 81.2% (0/39) |
| CNN of this project, third round, candidate B, run with seed 2 | 81.2% (170/142) | 78.4% (171/130) | 76.6% (153/141) | 76.0% (158/134) | 83.9% (182/140) | 83.3% (182/138) | 83.3% (0/40) | 83.3% (0/40) |
| CNN of this project, third round, candidate C | 73.7% (161/122) | 70.8% (159/113) | 89.1% (176/166) | 87.8% (178/159) | 85.9% (172/158) | 84.9% (172/154) | 60.4% (0/29) | 60.4% (0/29) |
| Manish18 ViT detector | 65.6% (80/172) | 65.1% (83/167) | 69.0% (104/161) | 71.9% (115/161) | 61.7% (88/149) | 62.0% (89/149) | 68.8% (0/33) | 68.8% (0/33) |
| umm-maybe Swin detector | 54.7% (164/46) | 53.1% (165/39) | 59.6% (161/68) | 61.5% (167/69) | 63.5% (167/77) | 63.3% (168/75) | 12.5% (0/6) | 10.4% (0/5) |
| prithivMLmods SigLIP 2 detector | 53.1% (101/103) | 52.9% (100/103) | 60.9% (114/120) | 60.9% (114/120) | 45.6% (76/99) | 45.6% (75/100) | 39.6% (0/19) | 39.6% (0/19) |

**Second round: held-out sets (scored once, after the decisions of the second round; development data of the third round)**

| Approach | Final test set (384) | Final test set (384), JPEG 90 | ChatGPT and Gemini set (168) | ChatGPT and Gemini set (168), JPEG 90 | Reddit set (120) | Reddit set (120), JPEG 90 |
|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 96.6% (190/181) |  | 85.1% (62/81) |  | 96.7% (59/57) |  |
| Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | 95.3% (188/178) | 90.1% (189/157) | 69.0% (71/45) | 62.5% (71/34) | 70.8% (59/26) | 70.8% (60/25) |
| Community Forensics detector, threshold tuned on the development images | 94.0% (183/178) | 88.5% (187/153) | 61.3% (70/33) | 58.3% (71/27) | 70.8% (58/27) | 70.0% (57/27) |
| Community Forensics detector | 92.2% (190/164) | 82.3% (190/126) | 57.7% (71/26) | 50.6% (71/14) | 64.2% (60/17) | 65.8% (60/19) |
| CNN of this project, setting 5, third run | 83.9% (173/149) | 83.6% (180/141) | 70.8% (60/59) | 66.7% (66/46) | 70.8% (56/29) | 69.2% (56/27) |
| CNN of this project, setting 5, second run | 83.6% (173/148) | 82.0% (181/134) | 67.9% (52/62) | 64.3% (60/48) | 70.0% (57/27) | 70.0% (57/27) |
| CNN of this project, setting 4, threshold tuned on the development images | 83.1% (164/155) | 78.4% (173/128) | 69.6% (47/70) | 69.0% (55/61) | 71.7% (56/30) | 70.0% (56/28) |
| CNN of this project, setting 5 before the threshold | 82.8% (165/153) | 80.2% (169/139) | 64.9% (52/57) | 63.1% (57/49) | 68.3% (56/26) | 68.3% (56/26) |
| CNN of this project, setting 4 (first round) | 82.3% (147/169) | 81.0% (156/155) | 73.2% (43/80) | 72.6% (50/72) | 79.2% (55/40) | 75.8% (55/36) |
| CNN of this project, setting 5 (published) | 81.5% (171/142) | 78.1% (174/126) | 68.5% (61/54) | 63.1% (63/43) | 68.3% (56/26) | 69.2% (57/26) |
| CNN of this project, first attempt (setting 1) | 77.6% (124/174) | 75.5% (131/159) | 59.5% (41/59) | 58.9% (47/52) | 70.0% (38/46) | 70.8% (39/46) |
| Manish18 ViT detector | 66.4% (87/168) | 63.3% (87/156) | 70.8% (42/77) | 72.0% (45/76) | 57.5% (22/47) | 57.5% (22/47) |
| umm-maybe Swin detector | 59.4% (165/63) | 56.2% (164/52) | 52.4% (59/29) | 54.2% (63/28) | 59.2% (53/18) | 60.0% (54/18) |
| prithivMLmods SigLIP 2 detector | 50.0% (102/90) | 49.7% (101/90) | 57.7% (45/52) | 57.1% (43/53) | 46.7% (26/30) | 46.7% (26/30) |

> [!NOTE]
> One of the five published detectors that were evaluated, the one that the public page uses, is left out of these tables; its predictions are in the database `results/results.db` and its result files in `results/`.
<!-- /results -->

### What the third round shows

The third round trained the CNN further on 4,920 new images from OpenFake v2.1 only (photographs from LAION, Pexels and photography subreddits; images of recent API generators and from AI subreddits), chose among three candidates by a rule committed in advance, and scored every approach once on four new held-out sets ([PROTOCOL.md](PROTOCOL.md)).

1. **Better on new kinds of image, not overall.** The combination of the third round (candidate B with the Community Forensics detector, weight 0.5) scores **82.9%** on ChatGPT and Gemini set 3 and Reddit set 3 together (768 images), against **73.7%** for the combination of the second round (102 images right only with the new one, 31 only with the old; exact McNemar test, p = 4.9e-10). On final test set 3, of the older kinds, it scores **92.4%** against **94.5%** (6 against 14; p = 0.12, not significant).
2. **The cost is more false alarms on photographs.** The new combination accepts 176 of the 192 photographs of final test set 3 (second round: 189) and 76 of the 96 ImageNet photographs of ChatGPT and Gemini set 3 (second round: 96).
3. **Forgetting is real.** Candidate C, trained from the base network on the new data only, was rejected on the development data because it lost the older generators. On the held-out sets it is the most accurate approach on ChatGPT and Gemini set 3 (89.1%) and Reddit set 3 (85.9%), and the least accurate CNN on final test set 3 (73.7%).
4. **Runs differ.** Of three training runs of candidate B, the published seed-42 run is the best on the new kinds (82.9%, 81.1%, 81.2% in the combination) and the lowest on final test set 3 (92.4%, 93.8%, 93.2%). The Seedream 5.0 control, a generator in no training data, varies most (75.0% to 85.4% for the combination) and has only 48 images.
5. **Limits.** The generators of ChatGPT and Gemini set 3 and the subreddits of Reddit set 3 are in the new training data, from the same platforms at other times, so these two sets measure new images of the kinds trained on and may flatter the round. The full notes, deviations included, are at the end of [COMPARISON.md](COMPARISON.md).

### What the second-round results show

1. **Images of known kinds (final test set, 384 images).** Claude Sonnet 5.5 **96.6%**, the combination **95.3%**, the published Community Forensics detector **92.2%**, the CNN of this project, setting 5 (published), **81.5%**, the four other published detectors 50.0% to 75.8%. The combination is significantly better than the Community Forensics detector as distributed (exact McNemar test, p = 0.004). Much of that gain comes from the decision threshold: with a threshold tuned on the same development images, that detector alone reaches 94.0%, and what the CNN adds beyond it is not significant (p = 0.302). The difference between the combination and Claude Sonnet 5.5 is not significant either (p = 0.458).
2. **JPEG re-compression.** The Community Forensics detector falls from 92.2% to 82.3%, because it then misses generated images (126 recognised instead of 164); the combination falls from 95.3% to 90.1%.
3. **New generators and images from Reddit.** On images from GPT Image 2, GPT Image 1.5, Nano Banana and Nano Banana Pro the combination scores 69.0% and on images taken from Reddit 70.8%: it accepts nearly every photograph but recognises fewer than half of the generated images. Claude Sonnet 5.5 is the most accurate on both sets (85.1% and 96.7%). No detector was good on every kind of image.
4. **An answer "generated" is more reliable than an answer "real".** When the combination called an image generated, it was right for 98% of those images on the final test set, 98% on the ChatGPT and Gemini set and 96% on the Reddit set. When it called an image real, it was right for 93%, 58% and 63%. On new kinds of image, "real" means only that no known trace was found.
5. **The second round did not improve the CNN itself.** The stronger shrinking (setting 5) and the tuned threshold were adopted by rules fixed in advance. On the final test set setting 5 (published), at 81.5%, cannot be told apart from setting 4 (first round), at 82.3% (p = 0.791). On the other two sets setting 4 (first round) scores higher: 73.2% against 68.5% on the ChatGPT and Gemini set (p = 0.291, not significant) and 79.2% against 68.3% on the Reddit set (p = 0.002). Setting 4 (first round) is kept in the model repository under the tag `round-1`: `pipeline("image-classification", model="mkk1135/ai-image-detector", revision="round-1")`.
6. **A small test set misleads.** An earlier CNN (setting 3) scored 97.9% on the pilot test set of 48 images and 80.7% on the 384 images of what is now the development test set.
7. **Variation between training runs.** Setting 5 was trained three times with the same code and data, differing only in the seed, and each run received its own threshold by the same rule. On the final test set the three runs score 81.5% (seed 42, published), 83.6% (seed 1) and 83.9% (seed 2): mean 83.0%, standard deviation 1.3 points. The chosen thresholds on the "AI" score differ widely: 0.69, 0.38 and 0.30. The published run was fixed in advance as the seed-42 run and is the least accurate of the three on the final test set.

> [!WARNING]
> The answer of any of these detectors is an indication, not proof.

## How the evaluation was done

1. **First round.** A pilot test set of 48 images was too small and too much like the data used for tuning: an early model scored far higher on it than on a test set of 384 images (point 6 above). All approaches were then scored once on that larger set (`results/main/`).
2. **Why a second round.** Those results showed where the fine-tuned CNN was weak, so any further change would have been tuned on that test set. It was demoted to the *development test set*. The check set and the development test set together (541 images) are called the *development images* in this README; every decision of the second round was taken on them.
3. **Second round.** Three new sets were drawn and held out: a final test set of 384 images, a set of images from the generators behind ChatGPT and Gemini, and a set of images as shared on Reddit.

   The decision rules (which training setting to adopt, how to choose the threshold, whether to combine detectors, which comparisons count) were committed in [PROTOCOL.md](PROTOCOL.md) before any of these images was scored. `decide.py` applies the rules to the results database. The decisions were committed in commit `7b8e05d`; every approach was then scored once on the three held-out sets, and nothing was changed afterwards.
4. **Third round.** The second round's held-out sets showed that the system was weak on new generators and on images as shared online, so all five scored sets became development data (1,213 images) and four new sets were drawn: final test set 3, ChatGPT and Gemini set 3, a Seedream 5.0 control and Reddit set 3.

   The training data (only new data, from OpenFake v2.1, 4,920 images; `make_train_set.py`, list in `results/round3/`), three candidates (A: setting 5 trained further; B: the same with the processing of a sharing platform; C: the base network trained on the new data only), the rule for adopting one, the held-out sets and the primary comparisons were committed in [PROTOCOL.md](PROTOCOL.md) (commit `4aa51fd`) before any image of the round was downloaded; the reduction of the training set to about 5,000 images at the student's request was committed before any download (`c2f4104`).

   The decisions were committed in `6a47cc0` before any held-out image was scored, and the final scoring in `5cb2054`. `decide.py round3` reproduces the decisions from the database.

| Set | Folder | Images | Role |
|:---|:---|---:|:---|
| Pilot test set | `results/` | 48 | first round: early scores |
| Check set | `results/check/` | 157 | choosing training settings; processing test; development in every round |
| Development test set | `results/main/` | 384 | first round: test. Second and third rounds: development |
| **Final test set** | `results/final/` | 384 | second round: held out, scored once. Third round: development |
| **ChatGPT and Gemini set** | `results/chat/` | 168 | second round: held out, scored once. Third round: development |
| **Reddit set** | `results/wild/` | 120 | second round: held out, scored once; labels may be noisy. Third round: development |
| **Final test set 3** | `results/final3/` | 384 | third round: held out, scored once |
| **ChatGPT and Gemini set 3** | `results/chat3/` | 384 | third round: held out, scored once |
| **Seedream 5.0 control** | `results/control3/` | 48 | third round: held out, scored once; generated images of a generator in no training data |
| **Reddit set 3** | `results/wild3/` | 384 | third round: held out, scored once; labels may be noisy |

Each folder holds the list of its images (`*_set.csv`, columns file, class, source, author_or_generator, licence, url), the metrics of every approach, its answer for every image and a confusion matrix. What the list records depends on where an image comes from:

- Photographs from Wikimedia Commons (pilot, check, development and final test sets) and from Unsplash and Open Images (development and final test sets): the photographer, the licence of that photograph and the page of that photograph.
- Generated images from the Community Forensics evaluation set (same sets): the generator, the licence of the dataset (CC BY-NC-SA 4.0) and the dataset file and position of the image.
- All images of the ChatGPT and Gemini set and of the Reddit set, which come from OpenFake and, for Nano Banana and Nano Banana Pro, from two community datasets: the dataset, the generator or the photo source (DOCCI, ImageNet, or the kind of subreddit), the licence of the dataset and the dataset file and row. The author and the rights of each of these images are not recorded; only the dataset's licence is.

The lists of the third-round sets also give the SHA-256 of every file and the folder in which the images were kept. The images themselves are not in this repository; `make_test_set.py` downloads them.

## Try the detector

**In the browser, with nothing to install:** open the [online demo](https://huggingface.co/spaces/mkk1135/ai-image-detector) and choose an image. The demo runs setting 5 (published) of the CNN alone, not the combination, inside your browser, so the image never leaves your device. The first visit downloads the detector (about 110 MB). Of an image larger than 1024 px the demo analyses only the central 1024 x 1024 px; run that way, setting 5 (published) scores 82.0% on the final test set (`results/final/crop1024/`), against 81.5% for whole images.

**On a public server, until 29 November 2026:** [https://35-169-35-59.sslip.io](https://35-169-35-59.sslip.io) is the page of `app.py` on an Amazon Web Services server, in a different mode from the local page described below. It runs the combined detector of this project and, at the same time, a published detector from the Hugging Face Hub (`second_opinion.py`; the model is set in `deploy/aws-setup.sh`), and shows as its only result that published detector's score for "AI-generated" in percent, for example "97% AI-generated". The answer of the combined detector is computed but not shown. When the published detector gives no result, the page says that the result is unavailable; it does not show the combined detector's answer instead. Apart from a logo (`assets/logo.svg`, also its browser icon), its title, the upload field and the result, the page shows only the sentence "Real or AI? Find out in seconds.", in large type.

The page can use Claude Opus 5.5 on the Claude API or on Amazon Bedrock in place of the published detector (AWS had not enabled Bedrock for the account by 20 November 2026); Claude gives a class, not a number.

> [!WARNING]
> Unlike the demo and the local page, this page sends the image to the server, where it is analysed and deleted within two hours, so do not upload private images.

The page was set up after the evaluation (`deploy/aws-setup.sh`) and was not itself evaluated. It analyses at most 20 images from one IP address in any 10 minutes. Its web server (Caddy) sends security headers (HSTS, Content-Security-Policy, X-Frame-Options, X-Content-Type-Options, Referrer-Policy and Permissions-Policy) and no Server header, and the server runs one fixed commit of this repository (`COMMIT` in `deploy/aws-setup.sh`).

**On your own computer:**

```bash
pip install -r requirements.txt
python app.py
```

> [!NOTE]
> `requirements.txt` gives minimum versions; `requirements.lock.txt` pins the exact environment (Python 3.13.2) in which every result was produced: `pip install -r requirements.lock.txt`.

Then open http://127.0.0.1:7860 and upload an image. The page runs the combination: setting 5 (published) of the CNN together with the published Community Forensics detector, which is downloaded on first use (87 MB). The image is analysed on your own computer and is not sent anywhere. The page shows the answer, the detector score for that answer with the remark that it is not a probability, and a warning that the decision is weak when that score is below 75%. Of an image wider or taller than 2048 px, both detectors see only the central 2048 x 2048 px.

**From a program:**

```python
from transformers import pipeline

detector = pipeline("image-classification", model="mkk1135/ai-image-detector")
print(detector("photo.jpg"))            # setting 5 (published) alone

from PIL import Image
from fusion import classifier           # fusion.py of this repository; reads results/decisions.json

combined = classifier()
print(combined(Image.open("photo.jpg").convert("RGB")))   # ("real" or "ai", score of that answer)
```

## The detector

`train.py` fine-tunes ConvNeXt-Tiny (`facebook/convnext-tiny-224`), a CNN pretrained on ImageNet, for three epochs on 36,562 images. A random 10% is held out for validation. Images flagged NSFW by Community Forensics are removed.

| Class | Source | Images |
|:---|:---|---:|
| Generated | [Community Forensics-Small](https://huggingface.co/datasets/OwensLab/CommunityForensics-Small) (Park and Owens, CVPR 2025): 377 community Stable Diffusion models | 14,592 |
| Generated | Community Forensics-Small: GLIDE, DeepFloyd IF, BigGAN | 2,992 |
| Real | Community Forensics-Small: COCO (4,831), VISION (1,153), Landscapes HQ (2,992), FFHQ (2,992) | 11,968 |
| Real | [DataSeeds.AI Sample Dataset](https://huggingface.co/datasets/Dataseeds/DataSeeds.AI-Sample-Dataset-DSD): peer-ranked photographs of varied subjects | 7,010 |

The model trains on 256 x 256 crops taken at the images' original resolution, and at inference it sees the whole image without resizing, because shrinking an image removes the fine traces that generators leave. Training images are randomly shrunk, flipped, blurred, given noise and JPEG-compressed up to twice, so that the model cannot use an image's processing history (generated images are uncompressed PNG, most real ones JPEG) or its sharpness as a shortcut.

Five training settings were tried; `train.py` contains settings 4 and 5 (`--setting 4`, `--setting 5`, the default):

| Setting | Change | Effect |
|:---:|:---|:---|
| 1 | whole image shrunk to 224 px | first attempt |
| 2 | crops at original resolution | kept fine detail, but took most clean photographs of subjects other than landscapes and faces for generated images |
| 3 | 7,010 photographs of varied subjects and more generated images added | far fewer clean photographs of other subjects taken for generated images; the setting changed more than one thing at once, so the cause was not isolated |
| 4 | random blur and noise | the model reads sharpness less as a sign of origin |
| 5 | stronger random shrinking: seven times in ten by any factor down to the crop size, instead of half the time by up to 2x | fewer small web photographs called generated and less sensitivity to blur and resizing, but fewer generated images caught |

After training, the decision threshold of setting 5 (0.69 on the "AI" score) was chosen on the development images and built into the published model. The combination (`fusion.py`) takes a weighted mean of the "AI" probabilities of this CNN and of the published Community Forensics detector, with weight 0.25 for the CNN and threshold 0.25, so that three quarters of its weight lie on the published Community Forensics detector; the weight and the threshold were chosen on the development images too.

```bash
python train.py --hub-id YOUR_HF_USERNAME/ai-image-detector    # needs a GPU; downloads about 19 GB
python train.py --seed 1 --out checkpoints/seed1               # a repeated run
python train.py --setting 4 --out checkpoints/s4               # setting 4 (first round)
python train.py --dry-run                                      # one epoch on 64 images, nothing uploaded
```

**Third round.** `train.py --data round3` trains on the new data of `make_train_set.py` instead; `--init` starts from other weights and `--social` adds the processing of a sharing platform to the augmentation (the exact commands of candidates A, B and C are in the docstring of `train.py` and in `results/training/r3_*.json`). Candidate B, adopted by the rule of the third round, is evaluated but not published: the model on the Hub is setting 5 of the second round (see the note on the public demonstration at the top).

The records of the published run (`results/training/s5_seed42.json`) and of setting 4 (`results/training/s4.json`) are in `results/training/`. Each holds the configuration of the run (command line, commit of the code, seed, setting, base model, epochs, batch size, learning rate, crop size, numbers of training and validation images) and its curves: training loss and learning rate every 25 steps, and validation accuracy and loss after each epoch. The run of setting 4 cannot be repeated exactly, because its classifier head was initialised before the seed was set. `python training_record.py TRAINER_STATE.json NAME --command ... --seed ... --setting ...` writes such a record from the `trainer_state.json` of a run.

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

<details>
<summary>The decisions themselves were taken on the development images only, with these commands:</summary>

```bash
python decide.py setting                      # rule 1: does setting 5 replace setting 4?
python decide.py threshold                    # rule 2; --model commfor or --model cnn_final for the secondary comparisons
python decide.py fusion                       # rule 3: combination with the Community Forensics detector
python apply_threshold.py checkpoints/s5_seed42/checkpoint-3087 cnn_s5_raw models/cnn_s5
python rethreshold.py commfor results/final results/final/jpeg90 results/chat results/chat/jpeg90 results/wild results/wild/jpeg90
```

</details>

> [!TIP]
> Set `DETECTOR_DEVICE=cuda` to run the detectors on a GPU.

### Which model files were scored

The result files name the model given to `evaluate.py`: a Hub model id or a local folder of this project. `results/provenance.json`, written by `python provenance.py`, records for every approach the SHA-256 of the weight file behind that name: for a published detector the exact Hub revision (commit) from which it was loaded, and for a model of this project the file in its folder together with every Hub revision that holds the identical file. The CNN scored as setting 5 (published) is byte for byte the file of commit `37da0ae` of `mkk1135/ai-image-detector`, and setting 4 (first round) the file of the tag `round-1` (commit `f62d9ff`). Models that were never published, such as the first attempt and setting 5 before the threshold, are identified by their hash only.

| Script | Purpose |
|:---|:---|
| `train.py` | fine-tunes the detector |
| `make_test_set.py` | builds the six evaluation sets and records every image |
| `evaluate.py` | scores a detector; `--jpeg 90` re-compresses first, `--crop 1024` analyses the centre only, `--compare NAME` runs McNemar tests against one model, `--chart` draws a bar chart |
| `commfor.py`, `fusion.py` | the published Community Forensics detector and its combination with the CNN |
| `blind_test.py` | blind test of a chat model |
| `diagnose.py`, `jpeg_quality.py` | how blur, noise, resizing and JPEG strength change a detector's answers |
| `check_overlap.py` | checks that no real evaluation photograph is a training image |
| `analyse_models.py`, `results_db.py` | model files and speed; SQLite database of all results (`results/results.db`) |
| `decide.py`, `apply_threshold.py`, `rethreshold.py` | the decisions of the protocol and their application |
| `provenance.py`, `training_record.py` | hashes and Hub revisions of the scored model files; public record of a training run |
| `app.py`, `space/`, `export_onnx.py` | the two interfaces; `space/check_demo.py` compares the browser demo with the Python detector |

### The results database

<details>
<summary><code>results/results.db</code> holds every prediction of every approach on every evaluation image: 39,060 predictions on 1,261 images.</summary>

| Table or view | Content |
|:---|:---|
| `images` | one row per evaluation image: set, class, source group, generator or photo category, author, licence, URL |
| `models` | one row per approach: architecture, number of weights, file size and format, input size, licence, speed |
| `predictions` | one row per model, set, condition and image: the answer and its score |
| `processing` | check-set images still classified correctly after blur, sharpening, downscaling or noise |
| `scores`, `by_group`, `by_generator` | views: accuracy per model, set and condition; correct answers per source and per generator |

```bash
sqlite3 results/results.db "select model, accuracy from scores where set_name = 'final' and condition = 'as downloaded' order by 2 desc"
```

</details>

A GitHub workflow (`.github/workflows/checks.yml`) runs on every push and pull request. It rebuilds this database with `python results_db.py`, which computes every figure of every result file again from the stored predictions (the four counts, accuracy, precision, recall, F1 and the 95% interval) and fails if any of them differs, and runs `python decide.py check`, which fails if the decisions recorded in `results/decisions.json` are not reproduced from the database.

## Blind test of a chat model

`blind_test.py prepare` writes shuffled copies of the images with neutral file names and no metadata, all re-encoded as JPEG with a longest side of at most 1568 px. The model is shown each copy with the question "Is this image a real photograph taken by a camera, or is it AI-generated / synthetic? Answer with only one word: Real or AI-Generated." and nothing else.

Claude Sonnet 5.5 (`claude-sonnet-5-5`) was run as separate agents, in a single pass with no retries. Each agent was given the folder with the anonymised copies of the whole set and told to open only the 40 to 48 copies of its batch and nothing else; this was an instruction, checked afterwards against the agents' transcripts (claude_blind_audit.txt in each results folder). Its raw answers are in each results folder (`claude_blind_answers.txt`), and the exact instruction each agent received is in `results/blind_prompt.txt`. Each agent judged the 40 to 48 images of its batch one after another in one context, whereas the detectors classify every image independently.

The answer keys were not in the folder given to the model: it held only the anonymised copies (`blind_test.py prepare` now writes the keys to a separate folder, `data/keys/`). An audit of the complete transcripts of the 15 runs on the held-out sets (`claude_blind_audit.txt` in each results folder) shows that every run opened exactly the images of its own batch, each once, with no other tool call than returning its answers, and that the returned answers are those recorded.

## Limits

- The final test set was held out, but it comes from the same kinds of source as the development images. The ChatGPT and Gemini set and the Reddit set show how the approaches do on other kinds of image.
- The sets are small. On the final test set the 95% confidence interval of the combination is 92.7 to 97.0%; the ChatGPT and Gemini set has 168 images and the Reddit set 120, so differences of a few points there are not meaningful.
- The images from GPT Image 2 and 1.5 are images that users published, and the two collections of Gemini images are community datasets with short descriptions. In the Reddit set the class of an image is the kind of subreddit it was posted in, so some labels may be wrong.
- **Very large images.** The CNN analyses images at their original resolution, and an image of more than 8 megapixels does not fit into the memory of the 8 GB GPU used. In the evaluation, of such an image the CNN analysed only the central part of at most 2048 x 2048 pixels (`fit` in `evaluate.py`); the published detectors shrink every image themselves and received the whole image, and in the combination only the CNN part was affected. The rule affected no image of the check set, the development test set or the final test set, one photograph of the ChatGPT and Gemini set and many of the Reddit images, most of them photographs (counts in [PROTOCOL.md](PROTOCOL.md)). The local interface is stricter: of any image wider or taller than 2048 px it analyses only the central 2048 x 2048 px, with both detectors. The browser demo analyses only the central 1024 x 1024 px.
- The fine-tuned CNN needs about 2.1 seconds per image on a laptop processor (images of 1.75 megapixels on average).
- Only one multimodal model was tested, once per image, and it saw JPEG copies with a longest side of at most 1568 px. Whether any test image was in the training data of Claude Sonnet 5.5 or of the published detectors is not known.
- The published model is one of three training runs of setting 5 (seed 42, fixed in advance). The two repeated runs (seeds 1 and 2) were trained and scored after the final scoring, with the same code and rules; on the final test set the three runs range from 81.5% to 83.9% (mean 83.0%, standard deviation 1.3 points), so differences of one or two points between single runs are within the variation between runs. Three runs give only a rough estimate of that variation.
- **Third round.** The generators of ChatGPT and Gemini set 3 and the subreddits of Reddit set 3 are in the third-round training data (from the same platforms, at other times), so these two sets measure new images of the kinds trained on, not unseen generators; final test set 3 and the Seedream 5.0 control (48 images) measure generators absent from all training data. The gain of the third round is on the new kinds of image; on final test set 3 the new system is 2.1 points less accurate (not significant) and accepts fewer photographs. On the sets of 384 images the three training runs of candidate B differ by up to 2.1 points, on the Seedream control by up to 10.4.
- Partly synthetic images, edited photographs, screenshots, video and deliberate evasion were not tested.
- The model may be used for non-commercial purposes only (CC BY-NC-SA 4.0, following its training data).
