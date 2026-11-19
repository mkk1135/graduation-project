# Comparison of the approaches

Written by `python results_db.py` from the database `results/results.db`, which holds every prediction of every approach on every evaluation image. Nothing here is typed by hand, except the notes at the end. The roles of the sets and the rules by which decisions were taken are in [PROTOCOL.md](PROTOCOL.md).

## 1. What each model is

From the files of each model (`python analyse_models.py`) and its page on the Hugging Face Hub.

| Approach | Architecture | Weights (million) | File (MB) | Safe file format | Input | Licence | Published | Downloads last month | Model card |
|---|---|---|---|---|---|---|---|---|---|
| CNN of this project, setting 4 | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | original resolution, not resized | cc-by-nc-sa-4.0 | 2026-11-18 | 46 | 929 words |
| CNN of this project, setting 3 | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | original resolution, not resized |  |  |  |  |
| CNN of this project, setting 2 | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | original resolution, not resized |  |  |  |  |
| CNN of this project, first attempt (setting 1) | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | resized to 224 x 224 px |  |  |  |  |
| Community Forensics detector | ViT-small (timm) | 21.8 | 87.3 | yes (safetensors) | resized to 384 x 384 px | mit | 2025-08-28 | 3193 | 31 words |
| Manish18 ViT detector | ViTForImageClassification | 85.8 | 343.2 | yes (safetensors) | resized to 224 x 224 px | none stated | 2026-08-08 | 9 | 735 words, 39 unfilled fields |
| umm-maybe Swin detector | SwinForImageClassification | 86.8 | 347.6 | **no (pickle)** | resized to 224 x 224 px | cc-by-4.0 | 2022-10-04 | 63600 | 286 words |
| Ateeqq SigLIP detector | SiglipForImageClassification | 92.9 | 371.6 | yes (safetensors) | resized to 224 x 224 px | apache-2.0 | 2025-03-30 | 11931 | 429 words |
| prithivMLmods SigLIP2 detector | SiglipForImageClassification | 92.9 | 371.6 | yes (safetensors) | resized to 224 x 224 px | apache-2.0 | 2024-03-17 | 1662 | 295 words |

What each publisher says about the training data:

- **CNN of this project, first attempt (setting 1):** 23,755 images: 8 shards of Community Forensics-Small; whole images shrunk to 224 px.
- **CNN of this project, setting 2:** 23,755 images: 8 shards of Community Forensics-Small; crops at original resolution.
- **CNN of this project, setting 3:** as setting 4, without blur and noise.
- **CNN of this project, setting 4:** 36,562 images: 10 shards of Community Forensics-Small and 7,010 DataSeeds photographs; crops at original resolution with blur and noise.
- **CNN of this project, setting 4, threshold tuned on development data:** as setting 4; only its decision threshold was chosen on the development images, for comparison with setting 5.
- **CNN of this project, setting 5 (published):** as setting 4, with stronger shrinking in the augmentation; decision threshold chosen on development data.
- **CNN of this project, setting 5 before the threshold:** as setting 5, with the default threshold of 50%.
- **CNN of this project, setting 5, second run:** as setting 5, another seed.
- **CNN of this project, setting 5, second run, before the threshold:** as setting 5, another seed, default threshold.
- **CNN of this project, setting 5, third run:** as setting 5, another seed.
- **CNN of this project, setting 5, third run, before the threshold:** as setting 5, another seed, default threshold.
- **Combination: CNN of this project and Community Forensics detector:** weighted mean of the two detectors' scores; weight and threshold chosen on development data.
- **Community Forensics detector, threshold tuned on development data:** as the published detector; only its decision threshold was chosen on the development images of this project.
- **Ateeqq SigLIP detector:** card: 60,000 AI-generated and 60,000 human images; card reports 99.2% on its own test split.
- **Community Forensics detector:** the authors' full Community Forensics data: 2.7 million images from 4,803 generators.
- **Manish18 ViT detector:** not stated: the model card is an unfilled template.
- **prithivMLmods SigLIP2 detector:** card: its own deepfake data; card reports 94.4% on its own test data.
- **umm-maybe Swin detector:** card: made in October 2022 for artistic images, 'not a deepfake photo detector'.
- **Claude Sonnet 5.5:** not a detector; general-purpose model of Anthropic.

## 2. Accuracy on every set

Percent of images classified correctly (real photographs correct / AI images correct). JPEG 90 = every image re-compressed first.

**Development data**

| Approach | Pilot test set (48) | Check set (157) | Check set (157), JPEG 90 | Development test set (384) | Development test set (384), JPEG 90 |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 97.9% (26/21) |  |  | 97.4% (190/184) |  |
| Combination: CNN of this project and Community Forensics detector |  | 92.4% (69/76) | 86.6% (69/67) | 93.8% (184/176) | 89.8% (186/159) |
| Community Forensics detector | 95.8% (26/20) | 86.0% (69/66) | 75.2% (69/49) | 89.6% (190/154) | 81.0% (190/121) |
| CNN of this project, setting 5 (published) |  | 88.5% (66/73) | 83.4% (66/65) | 82.5% (167/150) | 80.2% (170/138) |
| CNN of this project, setting 5 before the threshold |  | 87.3% (62/75) | 83.4% (63/68) | 81.0% (155/156) | 80.2% (161/147) |
| CNN of this project, setting 3 | 97.9% (25/22) | 88.5% (58/81) | 87.9% (60/78) | 80.7% (142/168) | 80.2% (151/157) |
| CNN of this project, setting 4 | 87.5% (21/21) | 89.2% (59/81) | 87.3% (60/77) | 80.0% (139/168) | 81.8% (150/164) |
| Ateeqq SigLIP detector | 87.5% (24/18) | 79.6% (56/69) | 79.6% (56/69) | 74.5% (150/136) | 74.7% (147/140) |
| CNN of this project, first attempt (setting 1) | 70.8% (14/20) | 79.6% (41/84) | 79.6% (42/83) | 74.0% (112/172) | 73.4% (121/161) |
| CNN of this project, setting 2 | 64.6% (10/21) | 76.4% (33/87) | 77.1% (35/86) | 66.4% (74/181) | 66.9% (78/179) |
| Manish18 ViT detector | 72.9% (16/19) | 76.4% (35/85) | 74.5% (35/82) | 61.5% (72/164) | 60.9% (73/161) |
| umm-maybe Swin detector | 66.7% (25/7) | 48.4% (59/17) | 48.4% (60/16) | 58.3% (173/51) | 58.1% (175/48) |
| prithivMLmods SigLIP2 detector | 47.9% (15/8) | 57.3% (44/46) | 56.7% (43/46) | 48.2% (97/88) | 47.9% (96/88) |

## 3. Development test set: images classified correctly, by source and generator

| Approach | Open Images (64) | Unsplash (64) | Wikimedia Commons (64) | Commercial generators (132) | Open-source generators (60) |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 63 | 63 | 64 | 131 | 53 |
| Combination: CNN of this project and Community Forensics detector | 61 | 61 | 62 | 117 | 59 |
| Community Forensics detector | 64 | 62 | 64 | 95 | 59 |
| CNN of this project, setting 5 (published) | 48 | 60 | 59 | 114 | 36 |
| CNN of this project, setting 5 before the threshold | 44 | 57 | 54 | 118 | 38 |
| CNN of this project, setting 3 | 30 | 55 | 57 | 122 | 46 |
| CNN of this project, setting 4 | 31 | 54 | 54 | 122 | 46 |
| Ateeqq SigLIP detector | 49 | 51 | 50 | 109 | 27 |
| CNN of this project, first attempt (setting 1) | 29 | 43 | 40 | 125 | 47 |
| CNN of this project, setting 2 | 17 | 29 | 28 | 132 | 49 |
| Manish18 ViT detector | 23 | 24 | 25 | 120 | 44 |
| umm-maybe Swin detector | 50 | 64 | 59 | 31 | 20 |
| prithivMLmods SigLIP2 detector | 34 | 28 | 35 | 60 | 28 |

| Images | Claude Sonnet 5.5 | Combination: CNN of this project and Community Forensics detector | Community Forensics detector | CNN of this project, setting 5 (published) | CNN of this project, setting 5 before the threshold | CNN of this project, setting 3 | CNN of this project, setting 4 | Ateeqq SigLIP detector | CNN of this project, first attempt (setting 1) | CNN of this project, setting 2 | Manish18 ViT detector | umm-maybe Swin detector | prithivMLmods SigLIP2 detector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real: Open Images (64) | 63 | 61 | 64 | 48 | 44 | 30 | 31 | 49 | 29 | 17 | 23 | 50 | 34 |
| Real: Unsplash (64) | 63 | 61 | 62 | 60 | 57 | 55 | 54 | 51 | 43 | 29 | 24 | 64 | 28 |
| Real: objects (18) | 18 | 18 | 18 | 16 | 12 | 15 | 16 | 14 | 9 | 8 | 4 | 17 | 9 |
| Real: buildings (10) | 10 | 9 | 10 | 8 | 8 | 9 | 6 | 9 | 4 | 2 | 5 | 9 | 4 |
| Real: landscapes (10) | 10 | 10 | 10 | 10 | 9 | 9 | 10 | 10 | 10 | 9 | 6 | 10 | 5 |
| Real: people (10) | 10 | 9 | 10 | 9 | 9 | 8 | 8 | 6 | 7 | 4 | 7 | 9 | 5 |
| Real: flowers (8) | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 6 | 5 | 1 | 1 | 8 | 7 |
| Real: food (4) | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 2 | 3 | 2 | 1 | 4 | 3 |
| Real: markets (4) | 4 | 4 | 4 | 4 | 4 | 4 | 2 | 3 | 2 | 2 | 1 | 2 | 2 |
| AI: Dalle2 (12) | 12 | 10 | 9 | 8 | 10 | 12 | 11 | 8 | 11 | 12 | 11 | 1 | 6 |
| AI: Dalle3 (12) | 12 | 12 | 11 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 6 | 4 |
| AI: FLUX-dev (12) | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 7 | 3 |
| AI: FLUX-schnell (12) | 12 | 11 | 11 | 10 | 10 | 10 | 11 | 12 | 11 | 12 | 12 | 2 | 8 |
| AI: Firefly_Image2 (12) | 12 | 8 | 7 | 7 | 7 | 7 | 8 | 5 | 9 | 12 | 9 | 1 | 6 |
| AI: Firefly_Image3 (12) | 11 | 7 | 4 | 8 | 9 | 9 | 9 | 11 | 12 | 12 | 11 | 2 | 4 |
| AI: IdeogramV1 (12) | 12 | 12 | 7 | 12 | 12 | 12 | 12 | 11 | 12 | 12 | 11 | 2 | 6 |
| AI: IdeogramV2 (12) | 12 | 12 | 8 | 12 | 12 | 12 | 12 | 7 | 12 | 12 | 10 | 2 | 5 |
| AI: Imagen3 (12) | 12 | 12 | 9 | 10 | 11 | 12 | 12 | 10 | 12 | 12 | 12 | 3 | 7 |
| AI: MidjourneyV5_2 (12) | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 10 | 2 | 8 |
| AI: MidjourneyV6_1 (12) | 12 | 9 | 5 | 11 | 11 | 12 | 11 | 9 | 10 | 12 | 10 | 3 | 3 |
| AI: DFGAN (6) | 5 | 6 | 6 | 1 | 1 | 3 | 3 | 1 | 3 | 1 | 3 | 6 | 5 |
| AI: DeciDiffusionV2 (6) | 6 | 6 | 6 | 4 | 5 | 6 | 4 | 4 | 5 | 6 | 4 | 0 | 3 |
| AI: GALIP (6) | 5 | 5 | 5 | 2 | 2 | 2 | 2 | 0 | 5 | 3 | 1 | 3 | 5 |
| AI: Hourglass (6) | 1 | 6 | 6 | 4 | 4 | 4 | 5 | 2 | 4 | 4 | 2 | 4 | 1 |
| AI: LCM_lora_sdv15 (6) | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 4 | 6 | 6 | 6 | 1 | 2 |
| AI: LCM_lora_sdxl (6) | 6 | 6 | 6 | 1 | 1 | 4 | 3 | 2 | 5 | 5 | 6 | 1 | 4 |
| AI: LCM_lora_ssd1b (6) | 6 | 6 | 6 | 3 | 4 | 4 | 5 | 3 | 5 | 6 | 6 | 1 | 3 |
| AI: kandinsky_2_2 (6) | 6 | 6 | 6 | 5 | 5 | 5 | 6 | 3 | 4 | 6 | 5 | 1 | 1 |
| AI: kvikontent_midjourney_v6 (6) | 6 | 6 | 6 | 4 | 4 | 6 | 6 | 2 | 4 | 6 | 5 | 3 | 3 |
| AI: stable_cascade (6) | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 0 | 1 |

## Sensitivity to image processing (check set)

Images of the check set (69 real, 88 AI) still classified correctly after one processing step, as real / AI. A detector that reads sharpness or cleanness instead of the way an image was made loses real photographs here.

| Approach | as downloaded | Gaussian blur, radius 0.6 | Gaussian blur, radius 1.0 | unsharp mask (sharpen) | downscaled to half size | Gaussian noise, sd 3 | Mean accuracy |
|---|---|---|---|---|---|---|---|
| CNN of this project, setting 5 before the threshold | 62 / 75 | 53 / 77 | 56 / 75 | 63 / 75 | 61 / 77 | 63 / 73 | 86.0% |
| CNN of this project, setting 4 | 59 / 81 | 40 / 81 | 45 / 78 | 62 / 80 | 41 / 86 | 63 / 78 | 84.3% |
| Community Forensics detector | 69 / 66 | 69 / 65 | 69 / 65 | 69 / 62 | 69 / 64 | 64 / 52 | 83.1% |
| CNN of this project, setting 3 | 58 / 81 | 34 / 82 | 23 / 81 | 64 / 81 | 50 / 83 | 32 / 84 | 79.9% |
| Ateeqq SigLIP detector | 56 / 69 | 56 / 69 | 56 / 70 | 57 / 69 | 56 / 69 | 56 / 69 | 79.8% |
| CNN of this project, first attempt (setting 1) | 41 / 84 | 36 / 84 | 32 / 87 | 43 / 84 | 41 / 84 | 35 / 84 | 78.0% |
| Manish18 ViT detector | 35 / 85 | 35 / 85 | 35 / 84 | 35 / 84 | 35 / 85 | 35 / 83 | 76.0% |
| CNN of this project, setting 2 | 33 / 87 | 15 / 87 | 2 / 88 | 43 / 87 | 34 / 87 | 27 / 88 | 72.0% |
| prithivMLmods SigLIP2 detector | 44 / 46 | 44 / 46 | 43 / 47 | 44 / 46 | 44 / 46 | 44 / 46 | 57.3% |
| umm-maybe Swin detector | 59 / 17 | 59 / 17 | 59 / 20 | 58 / 15 | 58 / 19 | 56 / 25 | 49.0% |

## Ranking quality, confidence and speed (development test set)

AUC = probability that a real photograph gets a higher 'real' score than a generated image (1.0 = perfect, 0.5 = guessing); it does not depend on the decision threshold. Seconds = time per image on this laptop's processor, without a GPU.

| Approach | AUC | Correct when the score is 0.9 or more | Correct when the score is below 0.75 | Seconds per image (mean) | Seconds (slowest image) |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | no scores |  |  |  |  |
| Combination: CNN of this project and Community Forensics detector | 0.971 | 97% of 258 | 79% of 61 |  |  |
| Community Forensics detector | 0.972 | 94% of 342 | 57% of 21 | 0.329 | 0.466 |
| CNN of this project, setting 5 (published) | 0.881 | 88% of 289 | 65% of 43 |  |  |
| CNN of this project, setting 5 before the threshold | 0.881 | 87% of 273 | 57% of 49 |  |  |
| CNN of this project, setting 3 | 0.906 | 88% of 250 | 62% of 56 | 3.568 | 9.603 |
| CNN of this project, setting 4 | 0.895 | 85% of 248 | 61% of 62 | 3.071 | 9.533 |
| Ateeqq SigLIP detector | 0.797 | 75% of 374 | 50% of 8 | 0.326 | 0.457 |
| CNN of this project, first attempt (setting 1) | 0.887 | 80% of 302 | 46% of 41 | 0.229 | 0.355 |
| CNN of this project, setting 2 | 0.880 | 71% of 294 | 57% of 56 | 3.606 | 9.504 |
| Manish18 ViT detector | 0.753 | 65% of 297 | 46% of 41 | 0.292 | 0.344 |
| umm-maybe Swin detector | 0.637 | 63% of 150 | 54% of 125 | 0.401 | 0.505 |
| prithivMLmods SigLIP2 detector | 0.490 | 70% of 43 | 46% of 250 | 0.295 | 0.484 |
