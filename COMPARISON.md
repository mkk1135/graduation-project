# Comparison of the approaches

Written by `python results_db.py` from the database `results/results.db`, which holds every prediction of every approach on every evaluation image. Nothing here is typed by hand, except the notes at the end. The roles of the sets and the rules by which decisions were taken are in [PROTOCOL.md](PROTOCOL.md).

## 1. What each model is

From the files of each model (`python analyse_models.py`) and its page on the Hugging Face Hub.

| Approach | Architecture | Weights (million) | File (MB) | Safe file format | Input | Licence | Published | Downloads last month | Model card |
|---|---|---|---|---|---|---|---|---|---|
| CNN of this project, setting 5 (published) | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | original resolution, not resized |  |  |  |  |
| CNN of this project, setting 4 (first round) | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | original resolution, not resized |  |  |  |  |
| CNN of this project, setting 3 | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | original resolution, not resized |  |  |  |  |
| CNN of this project, setting 2 | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | original resolution, not resized |  |  |  |  |
| CNN of this project, first attempt (setting 1) | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | resized to 224 x 224 px |  |  |  |  |
| Community Forensics detector | ViT-small (timm) | 21.8 | 87.3 | yes (safetensors) | resized to 384 x 384 px | mit | 2025-08-28 | 3193 | 31 words |
| Manish18 ViT detector | ViTForImageClassification | 85.8 | 343.2 | yes (safetensors) | resized to 224 x 224 px | none stated | 2026-08-08 | 9 | 735 words, 39 unfilled fields |
| umm-maybe Swin detector | SwinForImageClassification | 86.8 | 347.6 | **no (pickle)** | resized to 224 x 224 px | cc-by-4.0 | 2022-10-04 | 63600 | 286 words |
| Ateeqq SigLIP 2 detector | SiglipForImageClassification | 92.9 | 371.6 | yes (safetensors) | resized to 224 x 224 px | apache-2.0 | 2025-03-30 | 11931 | 429 words |
| prithivMLmods SigLIP 2 detector | SiglipForImageClassification | 92.9 | 371.6 | yes (safetensors) | resized to 224 x 224 px | apache-2.0 | 2024-03-17 | 1662 | 295 words |

What each publisher says about the training data:

- **CNN of this project, first attempt (setting 1):** 23,755 images: 8 shards of Community Forensics-Small; whole images shrunk to 224 px.
- **CNN of this project, setting 2:** 23,755 images: 8 shards of Community Forensics-Small; crops at original resolution.
- **CNN of this project, setting 3:** as setting 4, without blur and noise.
- **CNN of this project, setting 4 (first round):** 36,562 images: 10 shards of Community Forensics-Small and 7,010 DataSeeds photographs; crops at original resolution with blur and noise.
- **CNN of this project, setting 4, threshold tuned on the development images:** as setting 4; only its decision threshold was chosen on the development images, for comparison with setting 5.
- **CNN of this project, setting 5 (published):** as setting 4, with stronger shrinking in the augmentation; decision threshold chosen on development data.
- **CNN of this project, setting 5 before the threshold:** as setting 5, with the default threshold of 50%.
- **CNN of this project, setting 5, second run:** as setting 5, another seed.
- **CNN of this project, setting 5, second run, before the threshold:** as setting 5, another seed, default threshold.
- **CNN of this project, setting 5, third run:** as setting 5, another seed.
- **CNN of this project, setting 5, third run, before the threshold:** as setting 5, another seed, default threshold.
- **Combination: CNN of this project and Community Forensics detector:** weighted mean of the two detectors' scores; weight and threshold chosen on development data.
- **Community Forensics detector, threshold tuned on the development images:** as the published detector; only its decision threshold was chosen on the development images of this project.
- **Ateeqq SigLIP 2 detector:** card: 60,000 AI-generated and 60,000 human images; card reports 99.2% on its own test split.
- **Community Forensics detector:** the authors' full Community Forensics data: 2.7 million images from 4,803 generators.
- **Manish18 ViT detector:** not stated: the model card is an unfilled template.
- **prithivMLmods SigLIP 2 detector:** card: its own deepfake data; card reports 94.4% on its own test data.
- **umm-maybe Swin detector:** card: made in October 2022 for artistic images, 'not a deepfake photo detector'.
- **Claude Sonnet 5.5:** not a detector; general-purpose model of Anthropic.

## 2. Accuracy on every set

Percent of images classified correctly (real photographs correct / AI images correct). JPEG 90 = every image re-compressed first.

**Held-out sets (scored once, after all decisions)**

| Approach | Final test set (384) | Final test set (384), JPEG 90 | ChatGPT and Gemini set (168) | ChatGPT and Gemini set (168), JPEG 90 | Reddit set (120) | Reddit set (120), JPEG 90 |
|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 96.6% (190/181) |  | 85.1% (62/81) |  | 96.7% (59/57) |  |
| Combination: CNN of this project and Community Forensics detector | 95.3% (188/178) | 90.1% (189/157) | 69.0% (71/45) | 62.5% (71/34) | 70.8% (59/26) | 70.8% (60/25) |
| Community Forensics detector, threshold tuned on the development images | 94.0% (183/178) | 88.5% (187/153) | 61.3% (70/33) | 58.3% (71/27) | 70.8% (58/27) | 70.0% (57/27) |
| Community Forensics detector | 92.2% (190/164) | 82.3% (190/126) | 57.7% (71/26) | 50.6% (71/14) | 64.2% (60/17) | 65.8% (60/19) |
| CNN of this project, setting 5, third run | 83.9% (173/149) | 83.6% (180/141) | 70.8% (60/59) | 66.7% (66/46) | 70.8% (56/29) | 69.2% (56/27) |
| CNN of this project, setting 5, second run | 83.6% (173/148) | 82.0% (181/134) | 67.9% (52/62) | 64.3% (60/48) | 70.0% (57/27) | 70.0% (57/27) |
| CNN of this project, setting 4, threshold tuned on the development images | 83.1% (164/155) | 78.4% (173/128) | 69.6% (47/70) | 69.0% (55/61) | 71.7% (56/30) | 70.0% (56/28) |
| CNN of this project, setting 5 before the threshold | 82.8% (165/153) | 80.2% (169/139) | 64.9% (52/57) | 63.1% (57/49) | 68.3% (56/26) | 68.3% (56/26) |
| CNN of this project, setting 4 (first round) | 82.3% (147/169) | 81.0% (156/155) | 73.2% (43/80) | 72.6% (50/72) | 79.2% (55/40) | 75.8% (55/36) |
| CNN of this project, setting 5 (published) | 81.5% (171/142) | 78.1% (174/126) | 68.5% (61/54) | 63.1% (63/43) | 68.3% (56/26) | 69.2% (57/26) |
| CNN of this project, first attempt (setting 1) | 77.6% (124/174) | 75.5% (131/159) | 59.5% (41/59) | 58.9% (47/52) | 70.0% (38/46) | 70.8% (39/46) |
| Ateeqq SigLIP 2 detector | 75.8% (160/131) | 75.8% (159/132) | 82.1% (61/77) | 81.0% (58/78) | 83.3% (50/50) | 83.3% (50/50) |
| Manish18 ViT detector | 66.4% (87/168) | 63.3% (87/156) | 70.8% (42/77) | 72.0% (45/76) | 57.5% (22/47) | 57.5% (22/47) |
| umm-maybe Swin detector | 59.4% (165/63) | 56.2% (164/52) | 52.4% (59/29) | 54.2% (63/28) | 59.2% (53/18) | 60.0% (54/18) |
| prithivMLmods SigLIP 2 detector | 50.0% (102/90) | 49.7% (101/90) | 57.7% (45/52) | 57.1% (43/53) | 46.7% (26/30) | 46.7% (26/30) |

**Development data**

| Approach | Pilot test set (48) | Check set (157) | Check set (157), JPEG 90 | Development test set (384) | Development test set (384), JPEG 90 |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 97.9% (26/21) |  |  | 97.4% (190/184) |  |
| Combination: CNN of this project and Community Forensics detector |  | 92.4% (69/76) | 86.6% (69/67) | 93.8% (184/176) | 89.8% (186/159) |
| Community Forensics detector | 95.8% (26/20) | 86.0% (69/66) | 75.2% (69/49) | 89.6% (190/154) | 81.0% (190/121) |
| CNN of this project, setting 5, third run |  | 90.4% (66/76) | 86.0% (66/69) | 85.4% (168/160) | 82.0% (171/144) |
| CNN of this project, setting 5, second run |  | 87.3% (64/73) | 83.4% (64/67) | 84.1% (167/156) | 81.5% (170/143) |
| CNN of this project, setting 5 before the threshold |  | 87.3% (62/75) | 83.4% (63/68) | 81.0% (155/156) | 80.2% (161/147) |
| CNN of this project, setting 4 (first round) | 87.5% (21/21) | 89.2% (59/81) | 87.3% (60/77) | 79.9% (139/168) | 81.8% (150/164) |
| CNN of this project, setting 5 (published) |  | 88.5% (66/73) | 83.4% (66/65) | 82.6% (167/150) | 80.2% (170/138) |
| CNN of this project, first attempt (setting 1) | 70.8% (14/20) | 79.6% (41/84) | 79.6% (42/83) | 74.0% (112/172) | 73.4% (121/161) |
| Ateeqq SigLIP 2 detector | 87.5% (24/18) | 79.6% (56/69) | 79.6% (56/69) | 74.5% (150/136) | 74.7% (147/140) |
| Manish18 ViT detector | 72.9% (16/19) | 76.4% (35/85) | 74.5% (35/82) | 61.5% (72/164) | 60.9% (73/161) |
| umm-maybe Swin detector | 66.7% (25/7) | 48.4% (59/17) | 48.4% (60/16) | 58.3% (173/51) | 58.1% (175/48) |
| prithivMLmods SigLIP 2 detector | 47.9% (15/8) | 57.3% (44/46) | 56.7% (43/46) | 48.2% (97/88) | 47.9% (96/88) |
| CNN of this project, setting 5, second run, before the threshold |  | 87.3% (65/72) | 82.8% (66/64) | 83.6% (170/151) | 82.3% (177/139) |
| CNN of this project, setting 5, third run, before the threshold |  | 87.9% (67/71) | 83.4% (67/64) | 83.3% (173/147) | 79.4% (177/128) |
| CNN of this project, setting 3 | 97.9% (25/22) | 88.5% (58/81) | 87.9% (60/78) | 80.7% (142/168) | 80.2% (151/157) |
| CNN of this project, setting 2 | 64.6% (10/21) | 76.4% (33/87) | 77.1% (35/86) | 66.4% (74/181) | 66.9% (78/179) |

## 3. Final test set: images classified correctly, by source and generator

| Approach | Open Images (64) | Unsplash (64) | Wikimedia Commons (64) | Commercial generators (132) | Open-source generators (60) |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 64 | 62 | 64 | 131 | 50 |
| Combination: CNN of this project and Community Forensics detector | 62 | 62 | 64 | 118 | 60 |
| Community Forensics detector, threshold tuned on the development images | 63 | 58 | 62 | 118 | 60 |
| Community Forensics detector | 64 | 62 | 64 | 104 | 60 |
| CNN of this project, setting 5, third run | 50 | 62 | 61 | 113 | 36 |
| CNN of this project, setting 5, second run | 50 | 63 | 60 | 112 | 36 |
| CNN of this project, setting 4, threshold tuned on the development images | 42 | 61 | 61 | 116 | 39 |
| CNN of this project, setting 5 before the threshold | 50 | 61 | 54 | 111 | 42 |
| CNN of this project, setting 4 (first round) | 33 | 59 | 55 | 124 | 45 |
| CNN of this project, setting 5 (published) | 53 | 62 | 56 | 105 | 37 |
| CNN of this project, first attempt (setting 1) | 31 | 50 | 43 | 129 | 45 |
| Ateeqq SigLIP 2 detector | 47 | 54 | 59 | 103 | 28 |
| Manish18 ViT detector | 26 | 26 | 35 | 124 | 44 |
| umm-maybe Swin detector | 42 | 59 | 64 | 43 | 20 |
| prithivMLmods SigLIP 2 detector | 34 | 34 | 34 | 58 | 32 |

| Images | Claude Sonnet 5.5 | Combination: CNN of this project and Community Forensics detector | Community Forensics detector, threshold tuned on the development images | Community Forensics detector | CNN of this project, setting 5, third run | CNN of this project, setting 5, second run | CNN of this project, setting 4, threshold tuned on the development images | CNN of this project, setting 5 before the threshold | CNN of this project, setting 4 (first round) | CNN of this project, setting 5 (published) | CNN of this project, first attempt (setting 1) | Ateeqq SigLIP 2 detector | Manish18 ViT detector | umm-maybe Swin detector | prithivMLmods SigLIP 2 detector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real: Open Images (64) | 64 | 62 | 63 | 64 | 50 | 50 | 42 | 50 | 33 | 53 | 31 | 47 | 26 | 42 | 34 |
| Real: Unsplash (64) | 62 | 62 | 58 | 62 | 62 | 63 | 61 | 61 | 59 | 62 | 50 | 54 | 26 | 59 | 34 |
| Real: objects (17) | 17 | 17 | 15 | 17 | 16 | 15 | 15 | 14 | 13 | 15 | 8 | 14 | 8 | 17 | 10 |
| Real: sculptures (8) | 8 | 8 | 8 | 8 | 8 | 7 | 7 | 7 | 7 | 7 | 4 | 7 | 3 | 8 | 6 |
| Real: ships (8) | 8 | 8 | 8 | 8 | 7 | 8 | 8 | 7 | 7 | 7 | 4 | 8 | 6 | 8 | 7 |
| Real: trees (8) | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 7 | 7 | 7 | 8 | 8 | 2 | 8 | 0 |
| Real: vehicles (8) | 8 | 8 | 8 | 8 | 7 | 7 | 8 | 7 | 7 | 7 | 6 | 8 | 5 | 8 | 3 |
| Real: mountains (6) | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 5 | 5 | 5 | 6 | 5 | 5 | 6 | 5 |
| Real: castles (5) | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 3 | 5 | 0 |
| Real: churches (4) | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 2 | 4 | 3 | 2 | 4 | 3 | 4 | 3 |
| AI: Dalle2 (12) | 12 | 11 | 11 | 11 | 7 | 6 | 8 | 6 | 10 | 5 | 11 | 6 | 10 | 5 | 6 |
| AI: Dalle3 (12) | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 10 | 12 | 8 | 8 |
| AI: FLUX-dev (12) | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 11 | 12 | 7 | 3 |
| AI: FLUX-schnell (12) | 12 | 12 | 12 | 12 | 11 | 11 | 12 | 12 | 12 | 11 | 12 | 12 | 12 | 5 | 5 |
| AI: Firefly_Image2 (12) | 12 | 9 | 11 | 7 | 6 | 4 | 7 | 6 | 8 | 5 | 12 | 7 | 11 | 2 | 6 |
| AI: Firefly_Image3 (12) | 11 | 7 | 6 | 5 | 10 | 11 | 9 | 9 | 10 | 9 | 12 | 10 | 10 | 5 | 5 |
| AI: IdeogramV1 (12) | 12 | 11 | 11 | 10 | 12 | 12 | 12 | 12 | 12 | 11 | 11 | 10 | 12 | 3 | 7 |
| AI: IdeogramV2 (12) | 12 | 12 | 11 | 8 | 11 | 11 | 11 | 12 | 12 | 12 | 11 | 8 | 10 | 2 | 6 |
| AI: Imagen3 (12) | 12 | 11 | 11 | 9 | 10 | 11 | 10 | 10 | 12 | 9 | 12 | 10 | 12 | 1 | 4 |
| AI: MidjourneyV5_2 (12) | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 11 | 12 | 11 | 12 | 2 | 5 |
| AI: MidjourneyV6_1 (12) | 12 | 9 | 9 | 6 | 10 | 10 | 11 | 8 | 12 | 8 | 12 | 8 | 11 | 3 | 3 |
| AI: DFGAN (6) | 5 | 6 | 6 | 6 | 2 | 2 | 3 | 3 | 5 | 2 | 3 | 0 | 4 | 5 | 5 |
| AI: DeciDiffusionV2 (6) | 6 | 6 | 6 | 6 | 4 | 5 | 5 | 5 | 5 | 4 | 5 | 3 | 4 | 1 | 2 |
| AI: GALIP (6) | 2 | 6 | 6 | 6 | 1 | 0 | 1 | 1 | 1 | 1 | 4 | 0 | 0 | 2 | 5 |
| AI: Hourglass (6) | 1 | 6 | 6 | 6 | 5 | 5 | 5 | 5 | 5 | 5 | 3 | 2 | 3 | 2 | 3 |
| AI: LCM_lora_sdv15 (6) | 6 | 6 | 6 | 6 | 5 | 5 | 5 | 6 | 5 | 4 | 5 | 4 | 6 | 1 | 1 |
| AI: LCM_lora_sdxl (6) | 6 | 6 | 6 | 6 | 0 | 0 | 1 | 1 | 3 | 1 | 5 | 4 | 5 | 0 | 2 |
| AI: LCM_lora_ssd1b (6) | 6 | 6 | 6 | 6 | 2 | 3 | 2 | 4 | 3 | 3 | 6 | 4 | 5 | 3 | 3 |
| AI: kandinsky_2_2 (6) | 6 | 6 | 6 | 6 | 6 | 5 | 6 | 6 | 6 | 6 | 4 | 4 | 6 | 1 | 4 |
| AI: kvikontent_midjourney_v6 (6) | 6 | 6 | 6 | 6 | 5 | 5 | 5 | 5 | 6 | 5 | 4 | 4 | 5 | 4 | 5 |
| AI: stable_cascade (6) | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 3 | 6 | 1 | 2 |

## 4. ChatGPT and Gemini set: images classified correctly, by source and generator

| Approach | DOCCI (36) | ImageNet (36) | Commercial generators (96) |
|---|---|---|---|
| Claude Sonnet 5.5 | 31 | 31 | 81 |
| Ateeqq SigLIP 2 detector | 28 | 33 | 77 |
| CNN of this project, setting 4 (first round) | 34 | 9 | 80 |
| CNN of this project, setting 5, third run | 34 | 26 | 59 |
| Manish18 ViT detector | 23 | 19 | 77 |
| CNN of this project, setting 4, threshold tuned on the development images | 36 | 11 | 70 |
| Combination: CNN of this project and Community Forensics detector | 35 | 36 | 45 |
| CNN of this project, setting 5 (published) | 35 | 26 | 54 |
| CNN of this project, setting 5, second run | 35 | 17 | 62 |
| CNN of this project, setting 5 before the threshold | 35 | 17 | 57 |
| Community Forensics detector, threshold tuned on the development images | 34 | 36 | 33 |
| CNN of this project, first attempt (setting 1) | 27 | 14 | 59 |
| Community Forensics detector | 35 | 36 | 26 |
| prithivMLmods SigLIP 2 detector | 15 | 30 | 52 |
| umm-maybe Swin detector | 32 | 27 | 29 |

| Images | Claude Sonnet 5.5 | Ateeqq SigLIP 2 detector | CNN of this project, setting 4 (first round) | CNN of this project, setting 5, third run | Manish18 ViT detector | CNN of this project, setting 4, threshold tuned on the development images | Combination: CNN of this project and Community Forensics detector | CNN of this project, setting 5 (published) | CNN of this project, setting 5, second run | CNN of this project, setting 5 before the threshold | Community Forensics detector, threshold tuned on the development images | CNN of this project, first attempt (setting 1) | Community Forensics detector | prithivMLmods SigLIP 2 detector | umm-maybe Swin detector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real: DOCCI (36) | 31 | 28 | 34 | 34 | 23 | 36 | 35 | 35 | 35 | 35 | 34 | 27 | 35 | 15 | 32 |
| Real: ImageNet (36) | 31 | 33 | 9 | 26 | 19 | 11 | 36 | 26 | 17 | 17 | 36 | 14 | 36 | 30 | 27 |
| AI: gpt-image-1.5 (24) | 19 | 17 | 21 | 11 | 20 | 16 | 13 | 10 | 11 | 11 | 11 | 15 | 8 | 11 | 11 |
| AI: gpt-image-2 (24) | 21 | 17 | 18 | 14 | 19 | 16 | 5 | 12 | 16 | 13 | 2 | 14 | 2 | 8 | 6 |
| AI: nano-banana (24) | 21 | 21 | 24 | 22 | 23 | 23 | 21 | 23 | 23 | 24 | 17 | 16 | 13 | 18 | 3 |
| AI: nano-banana-pro (24) | 20 | 22 | 17 | 12 | 15 | 15 | 6 | 9 | 12 | 9 | 3 | 14 | 3 | 15 | 9 |

## 5. Reddit set: images classified correctly, by source and generator

| Images | Claude Sonnet 5.5 | Ateeqq SigLIP 2 detector | CNN of this project, setting 4 (first round) | CNN of this project, setting 4, threshold tuned on the development images | CNN of this project, setting 5, third run | Community Forensics detector, threshold tuned on the development images | Combination: CNN of this project and Community Forensics detector | CNN of this project, first attempt (setting 1) | CNN of this project, setting 5, second run | CNN of this project, setting 5 (published) | CNN of this project, setting 5 before the threshold | Community Forensics detector | umm-maybe Swin detector | Manish18 ViT detector | prithivMLmods SigLIP 2 detector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real: Reddit, photography subreddits (60) | 59 | 50 | 55 | 56 | 56 | 58 | 59 | 38 | 57 | 56 | 56 | 60 | 53 | 22 | 26 |
| AI: Reddit, AI subreddits (60) | 57 | 50 | 40 | 30 | 29 | 27 | 26 | 46 | 27 | 26 | 26 | 17 | 18 | 47 | 30 |

## 6. Development test set: images classified correctly, by source and generator

| Approach | Open Images (64) | Unsplash (64) | Wikimedia Commons (64) | Commercial generators (132) | Open-source generators (60) |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 63 | 63 | 64 | 131 | 53 |
| Combination: CNN of this project and Community Forensics detector | 61 | 61 | 62 | 117 | 59 |
| Community Forensics detector | 64 | 62 | 64 | 95 | 59 |
| CNN of this project, setting 5, third run | 47 | 61 | 60 | 120 | 40 |
| CNN of this project, setting 5, second run | 46 | 61 | 60 | 119 | 37 |
| CNN of this project, setting 5, second run, before the threshold | 48 | 62 | 60 | 117 | 34 |
| CNN of this project, setting 5, third run, before the threshold | 50 | 62 | 61 | 111 | 36 |
| CNN of this project, setting 5 (published) | 48 | 60 | 59 | 114 | 36 |
| CNN of this project, setting 5 before the threshold | 44 | 57 | 54 | 118 | 38 |
| CNN of this project, setting 3 | 30 | 55 | 57 | 122 | 46 |
| CNN of this project, setting 4 (first round) | 31 | 54 | 54 | 122 | 46 |
| Ateeqq SigLIP 2 detector | 49 | 51 | 50 | 109 | 27 |
| CNN of this project, first attempt (setting 1) | 29 | 43 | 40 | 125 | 47 |
| CNN of this project, setting 2 | 17 | 29 | 28 | 132 | 49 |
| Manish18 ViT detector | 23 | 24 | 25 | 120 | 44 |
| umm-maybe Swin detector | 50 | 64 | 59 | 31 | 20 |
| prithivMLmods SigLIP 2 detector | 34 | 28 | 35 | 60 | 28 |

| Images | Claude Sonnet 5.5 | Combination: CNN of this project and Community Forensics detector | Community Forensics detector | CNN of this project, setting 5, third run | CNN of this project, setting 5, second run | CNN of this project, setting 5, second run, before the threshold | CNN of this project, setting 5, third run, before the threshold | CNN of this project, setting 5 (published) | CNN of this project, setting 5 before the threshold | CNN of this project, setting 3 | CNN of this project, setting 4 (first round) | Ateeqq SigLIP 2 detector | CNN of this project, first attempt (setting 1) | CNN of this project, setting 2 | Manish18 ViT detector | umm-maybe Swin detector | prithivMLmods SigLIP 2 detector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real: Open Images (64) | 63 | 61 | 64 | 47 | 46 | 48 | 50 | 48 | 44 | 30 | 31 | 49 | 29 | 17 | 23 | 50 | 34 |
| Real: Unsplash (64) | 63 | 61 | 62 | 61 | 61 | 62 | 62 | 60 | 57 | 55 | 54 | 51 | 43 | 29 | 24 | 64 | 28 |
| Real: objects (18) | 18 | 18 | 18 | 16 | 16 | 16 | 16 | 16 | 12 | 15 | 16 | 14 | 9 | 8 | 4 | 17 | 9 |
| Real: buildings (10) | 10 | 9 | 10 | 9 | 9 | 9 | 10 | 8 | 8 | 9 | 6 | 9 | 4 | 2 | 5 | 9 | 4 |
| Real: landscapes (10) | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 9 | 9 | 10 | 10 | 10 | 9 | 6 | 10 | 5 |
| Real: people (10) | 10 | 9 | 10 | 9 | 9 | 9 | 9 | 9 | 9 | 8 | 8 | 6 | 7 | 4 | 7 | 9 | 5 |
| Real: flowers (8) | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 6 | 5 | 1 | 1 | 8 | 7 |
| Real: food (4) | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 2 | 3 | 2 | 1 | 4 | 3 |
| Real: markets (4) | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 2 | 3 | 2 | 2 | 1 | 2 | 2 |
| AI: Dalle2 (12) | 12 | 10 | 9 | 11 | 11 | 11 | 10 | 8 | 10 | 12 | 11 | 8 | 11 | 12 | 11 | 1 | 6 |
| AI: Dalle3 (12) | 12 | 12 | 11 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 6 | 4 |
| AI: FLUX-dev (12) | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 7 | 3 |
| AI: FLUX-schnell (12) | 12 | 11 | 11 | 11 | 10 | 10 | 11 | 10 | 10 | 10 | 11 | 12 | 11 | 12 | 12 | 2 | 8 |
| AI: Firefly_Image2 (12) | 12 | 8 | 7 | 7 | 7 | 5 | 4 | 7 | 7 | 7 | 8 | 5 | 9 | 12 | 9 | 1 | 6 |
| AI: Firefly_Image3 (12) | 11 | 7 | 4 | 9 | 9 | 9 | 7 | 8 | 9 | 9 | 9 | 11 | 12 | 12 | 11 | 2 | 4 |
| AI: IdeogramV1 (12) | 12 | 12 | 7 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 11 | 12 | 12 | 11 | 2 | 6 |
| AI: IdeogramV2 (12) | 12 | 12 | 8 | 11 | 11 | 11 | 10 | 12 | 12 | 12 | 12 | 7 | 12 | 12 | 10 | 2 | 5 |
| AI: Imagen3 (12) | 12 | 12 | 9 | 12 | 12 | 12 | 10 | 10 | 11 | 12 | 12 | 10 | 12 | 12 | 12 | 3 | 7 |
| AI: MidjourneyV5_2 (12) | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 10 | 2 | 8 |
| AI: MidjourneyV6_1 (12) | 12 | 9 | 5 | 11 | 11 | 11 | 11 | 11 | 11 | 12 | 11 | 9 | 10 | 12 | 10 | 3 | 3 |
| AI: DFGAN (6) | 5 | 6 | 6 | 1 | 1 | 0 | 1 | 1 | 1 | 3 | 3 | 1 | 3 | 1 | 3 | 6 | 5 |
| AI: DeciDiffusionV2 (6) | 6 | 6 | 6 | 6 | 6 | 5 | 5 | 4 | 5 | 6 | 4 | 4 | 5 | 6 | 4 | 0 | 3 |
| AI: GALIP (6) | 5 | 5 | 5 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 0 | 5 | 3 | 1 | 3 | 5 |
| AI: Hourglass (6) | 1 | 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 5 | 2 | 4 | 4 | 2 | 4 | 1 |
| AI: LCM_lora_sdv15 (6) | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 4 | 6 | 6 | 6 | 1 | 2 |
| AI: LCM_lora_sdxl (6) | 6 | 6 | 6 | 3 | 1 | 0 | 0 | 1 | 1 | 4 | 3 | 2 | 5 | 5 | 6 | 1 | 4 |
| AI: LCM_lora_ssd1b (6) | 6 | 6 | 6 | 3 | 2 | 2 | 3 | 3 | 4 | 4 | 5 | 3 | 5 | 6 | 6 | 1 | 3 |
| AI: kandinsky_2_2 (6) | 6 | 6 | 6 | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 6 | 3 | 4 | 6 | 5 | 1 | 1 |
| AI: kvikontent_midjourney_v6 (6) | 6 | 6 | 6 | 4 | 4 | 4 | 4 | 4 | 4 | 6 | 6 | 2 | 4 | 6 | 5 | 3 | 3 |
| AI: stable_cascade (6) | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 0 | 1 |

## Sensitivity to image processing (check set)

Images of the check set (69 real, 88 AI) still classified correctly after one processing step, as real / AI. A detector that reads sharpness or cleanness instead of the way an image was made loses real photographs here.

| Approach | as downloaded | Gaussian blur, radius 0.6 | Gaussian blur, radius 1.0 | unsharp mask (sharpen) | downscaled to half size | Gaussian noise, sd 3 | Mean accuracy |
|---|---|---|---|---|---|---|---|
| CNN of this project, setting 4, threshold tuned on the development images | 66 / 79 | 53 / 78 | 64 / 75 | 67 / 77 | 49 / 85 | 67 / 73 | 88.4% |
| CNN of this project, setting 5 (published) | 66 / 73 | 58 / 73 | 63 / 69 | 67 / 72 | 65 / 76 | 67 / 67 | 86.6% |
| CNN of this project, setting 5 before the threshold | 62 / 75 | 53 / 77 | 56 / 75 | 63 / 75 | 61 / 77 | 63 / 73 | 86.0% |
| CNN of this project, setting 4 (first round) | 59 / 81 | 40 / 81 | 45 / 78 | 62 / 80 | 41 / 86 | 63 / 78 | 84.3% |
| Community Forensics detector | 69 / 66 | 69 / 65 | 69 / 65 | 69 / 62 | 69 / 64 | 64 / 52 | 83.1% |
| CNN of this project, setting 3 | 58 / 81 | 34 / 82 | 23 / 81 | 64 / 81 | 50 / 83 | 32 / 84 | 79.9% |
| Ateeqq SigLIP 2 detector | 56 / 69 | 56 / 69 | 56 / 70 | 57 / 69 | 56 / 69 | 56 / 69 | 79.8% |
| CNN of this project, first attempt (setting 1) | 41 / 84 | 36 / 84 | 32 / 87 | 43 / 84 | 41 / 84 | 35 / 84 | 78.0% |
| Manish18 ViT detector | 35 / 85 | 35 / 85 | 35 / 84 | 35 / 84 | 35 / 85 | 35 / 83 | 76.0% |
| CNN of this project, setting 2 | 33 / 87 | 15 / 87 | 2 / 88 | 43 / 87 | 34 / 87 | 27 / 88 | 72.0% |
| prithivMLmods SigLIP 2 detector | 44 / 46 | 44 / 46 | 43 / 47 | 44 / 46 | 44 / 46 | 44 / 46 | 57.3% |
| umm-maybe Swin detector | 59 / 17 | 59 / 17 | 59 / 20 | 58 / 15 | 58 / 19 | 56 / 25 | 49.0% |

## Ranking quality, confidence and speed (final test set)

AUC = probability that a real photograph gets a higher 'real' score than a generated image (1.0 = perfect, 0.5 = guessing); it does not depend on the decision threshold. Seconds = time per image on this laptop's processor, without a GPU.

| Approach | AUC | Correct when the score is 0.9 or more | Correct when the score is below 0.75 | Seconds per image (mean) | Seconds (slowest image) |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | no scores |  |  |  |  |
| Combination: CNN of this project and Community Forensics detector | 0.986 | 99% of 258 | 79% of 53 |  |  |
| Community Forensics detector, threshold tuned on the development images | 0.988 | 96% of 341 | 71% of 31 |  |  |
| Community Forensics detector | 0.988 | 96% of 341 | 47% of 17 | 0.202 | 0.238 |
| CNN of this project, setting 5, third run | 0.899 | 93% of 217 | 58% of 72 |  |  |
| CNN of this project, setting 5, second run | 0.901 | 91% of 273 | 60% of 52 |  |  |
| CNN of this project, setting 4, threshold tuned on the development images | 0.907 | 89% of 247 | 67% of 76 |  |  |
| CNN of this project, setting 5 before the threshold | 0.888 | 90% of 244 | 71% of 59 |  |  |
| CNN of this project, setting 4 (first round) | 0.907 | 89% of 247 | 69% of 65 | 2.058 | 5.417 |
| CNN of this project, setting 5 (published) | 0.888 | 88% of 268 | 55% of 60 | 2.091 | 5.602 |
| CNN of this project, first attempt (setting 1) | 0.891 | 83% of 290 | 62% of 45 | 0.112 | 0.22 |
| Ateeqq SigLIP 2 detector | 0.810 | 76% of 376 | 67% of 3 | 0.301 | 0.38 |
| Manish18 ViT detector | 0.791 | 69% of 289 | 58% of 48 | 0.247 | 0.301 |
| umm-maybe Swin detector | 0.644 | 66% of 132 | 57% of 132 | 0.288 | 0.399 |
| prithivMLmods SigLIP 2 detector | 0.493 | 61% of 28 | 49% of 253 | 0.299 | 0.369 |

## What the comparison shows

Notes on the tables above. Every number in them is filled in from the results database. Statistical tests are exact McNemar tests on the same images; no correction for multiple comparisons was applied, and only the two primary comparisons were fixed in advance in [PROTOCOL.md](PROTOCOL.md). The *development images* are the check set and the development test set together (541 images); every decision was taken on them. The two versions of the project's CNN are called setting 4 (first round), with the default threshold, and setting 5 (published), with the threshold tuned on the development images.

1. **Ranking on the final test set (384 images).** Claude Sonnet 5.5 96.6%, the combination 95.3%, the Community Forensics detector 92.2%, setting 5 (published) 81.5%, the four other published detectors 50.0% to 75.8%. Primary comparisons: the combination is more accurate than the Community Forensics detector as distributed (p = 0.004); setting 5 (published) cannot be told apart from setting 4 (first round) (82.3%, p = 0.791).
2. **Secondary comparison: the published detector with a tuned threshold.** With its threshold chosen on the development images by the same rule as the project's own detectors, the Community Forensics detector reaches 94.0%. The combination is not significantly better than that (p = 0.302). Much of the advantage of the combination over the detector as distributed therefore comes from the threshold; what the CNN adds beyond it is not significant.
3. **JPEG re-compression at quality 90.** The Community Forensics detector falls from 92.2% to 82.3% (tuned threshold: 88.5%), the combination from 95.3% to 90.1%, setting 5 (published) from 81.5% to 78.1%. After re-compression the combination is clearly better than the detector as distributed (p < 0.001), but not than the tuned one (p = 0.238).
4. **ChatGPT and Gemini set (168 images).** Claude Sonnet 5.5 is the most accurate (85.1%), followed by the Ateeqq detector (82.1%). The combination scores 69.0%: it accepts 71 photographs and misses 1, but recognises only 45 generated images and misses 51. The Community Forensics detector scores 57.7% and recognises 26 generated images.
5. **Reddit set (120 images).** Claude Sonnet 5.5 is again the most accurate (96.7%), followed by the Ateeqq detector (83.3%) and setting 4 (first round) (79.2%). The combination scores 70.8%: it accepts 59 photographs and misses 1, but recognises only 26 generated images and misses 34. No detector was the best on all three held-out sets.
6. **Training setting and threshold, separated.** Accuracy on the final test set, the ChatGPT and Gemini set and the Reddit set: setting 4 (first round) 82.3%, 73.2% and 79.2%; setting 4 with a tuned threshold 83.1%, 69.6% and 71.7%; setting 5 with the default threshold 82.8%, 64.9% and 68.3%; setting 5 (published) 81.5%, 68.5% and 68.3%. On the final test set the four are close (setting 5 against setting 4, both at the default threshold: p = 0.888). On the two other sets setting 4 (first round) is the most accurate of the four, and setting 5 at the default threshold is less accurate than it (p = 0.020 and p = 0.002). On these two sets the change of training setting lowered accuracy; the tuned threshold lowered that of setting 4 as well, and changed that of setting 5 little.
7. **An answer "generated" was more reliable than an answer "real".** When the combination called an image generated, it was right for 98% of those images on the final test set, 98% on the ChatGPT and Gemini set and 96% on the Reddit set. When it called an image real, it was right for 93%, 58% and 63%. On new kinds of image, "real" means only that no known trace was found.
8. **Variation between training runs.** Setting 5 was trained three times, differing only in the seed, each run with its own threshold chosen by the same rule (0.69, 0.38 and 0.30 on the "AI" score). On the final test set the runs score 81.5% (seed 42, published), 83.6% (seed 1) and 83.9% (seed 2): mean 83.0%, standard deviation 1.3 points. The published run was fixed in advance and is the least accurate of the three there.
9. **Limits.** The sets are small: the 95% confidence interval of the combination's accuracy is 92.7 to 97.0% on the final test set, 61.7 to 75.5% on the ChatGPT and Gemini set and 62.2 to 78.2% on the Reddit set. In the Reddit set the class of an image is the kind of subreddit it was posted in, so some labels may be wrong. Claude Sonnet 5.5 was asked once per image and saw JPEG copies; whether any test image was in its training data, or in that of the published detectors, is not known. Three training runs give only a rough estimate of the variation between runs (standard deviation 1.3 points on the final test set), so differences of one or two points between single runs of the CNN are not meaningful.
