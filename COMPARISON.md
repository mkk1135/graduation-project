# Comparison of the approaches

Written by `python results_db.py` from the database `results/results.db`, which holds every prediction of every approach on every evaluation image. Nothing here is typed by hand, except the notes at the end.

## 1. What each model is

From the files of each model (`python analyse_models.py`) and its page on the Hugging Face Hub.

| Approach | Architecture | Weights (million) | File (MB) | Safe file format | Input | Licence | Published | Downloads last month | Model card |
|---|---|---|---|---|---|---|---|---|---|
| CNN of this project, final (setting 4) | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | original resolution, not resized | cc-by-nc-sa-4.0 | 2026-11-18 | 46 | 929 words |
| CNN of this project, setting 3 | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | original resolution, not resized |  |  |  |  |
| CNN of this project, setting 2 | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | original resolution, not resized |  |  |  |  |
| CNN of this project, first attempt (setting 1) | ConvNextForImageClassification | 27.8 | 111.3 | yes (safetensors) | resized to 224 x 224 px |  |  |  |  |
| Community Forensics detector | ViT-small (timm) | 21.8 | 87.3 | yes (safetensors) | resized to 384 x 384 px | mit | 2025-08-28 | 3193 | 31 words |
| Manish18 ViT detector | ViTForImageClassification | 85.8 | 343.2 | yes (safetensors) | resized to 224 x 224 px | none stated | 2026-08-08 | 9 | 735 words, 39 unfilled fields |
| umm-maybe Swin detector | SwinForImageClassification | 86.8 | 347.6 | **no (pickle)** | resized to 224 x 224 px | cc-by-4.0 | 2022-10-04 | 63600 | 286 words |
| Ateeqq SigLIP detector | SiglipForImageClassification | 92.9 | 371.6 | yes (safetensors) | resized to 224 x 224 px | apache-2.0 | 2025-03-30 | 11931 | 429 words |
| prithivMLmods SigLIP2 detector | SiglipForImageClassification | 92.9 | 371.6 | yes (safetensors) | resized to 224 x 224 px | apache-2.0 | 2024-03-17 | 1662 | 295 words |
| Claude Sonnet 5.5 |  |  |  |  |  |  |  |  |  |

What each publisher says about the training data:

- **CNN of this project, final (setting 4):** 36,562 images: 10 shards of Community Forensics-Small and 7,010 DataSeeds photographs; crops at original resolution with blur and noise.
- **CNN of this project, first attempt (setting 1):** 23,755 images: 8 shards of Community Forensics-Small; whole images shrunk to 224 px.
- **CNN of this project, setting 2:** 23,755 images: 8 shards of Community Forensics-Small; crops at original resolution.
- **CNN of this project, setting 3:** as the final model, without blur and noise.
- **Ateeqq SigLIP detector:** card: 60,000 AI-generated and 60,000 human images; card reports 99.2% on its own test split.
- **Community Forensics detector:** the authors' full Community Forensics data: 2.7 million images from 4,803 generators.
- **Manish18 ViT detector:** not stated: the model card is an unfilled template.
- **prithivMLmods SigLIP2 detector:** card: its own deepfake data; card reports 94.4% on its own test data.
- **umm-maybe Swin detector:** card: made in October 2022 for artistic images, 'not a deepfake photo detector'.
- **Claude Sonnet 5.5:** not a detector; general-purpose model of Anthropic.

## 2. Accuracy on every set

Percent of images classified correctly (real photographs correct / AI images correct). JPEG 90 = every image re-compressed first.

| Approach | Pilot (48) | Check (157) | Check, JPEG 90 | Main (384) | Main, JPEG 90 | ChatGPT and Gemini set (168) | ChatGPT and Gemini set, JPEG 90 |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 97.9% (26/21) | not measured | not measured | 97.4% (190/184) | not measured | not measured | not measured |
| Community Forensics detector | 95.8% (26/20) | 86.0% (69/66) | 75.2% (69/49) | 89.6% (190/154) | 81.0% (190/121) | not measured | not measured |
| CNN of this project, setting 3 | 97.9% (25/22) | 88.5% (58/81) | 87.9% (60/78) | 80.7% (142/168) | 80.2% (151/157) | not measured | not measured |
| CNN of this project, final (setting 4) | 87.5% (21/21) | 89.2% (59/81) | 87.3% (60/77) | 80.0% (139/168) | 81.8% (150/164) | not measured | not measured |
| Ateeqq SigLIP detector | 87.5% (24/18) | 79.6% (56/69) | 79.6% (56/69) | 74.5% (150/136) | 74.7% (147/140) | not measured | not measured |
| CNN of this project, first attempt (setting 1) | 70.8% (14/20) | 79.6% (41/84) | 79.6% (42/83) | 74.0% (112/172) | 73.4% (121/161) | not measured | not measured |
| CNN of this project, setting 2 | 64.6% (10/21) | 76.4% (33/87) | 77.1% (35/86) | 66.4% (74/181) | 66.9% (78/179) | not measured | not measured |
| Manish18 ViT detector | 72.9% (16/19) | 76.4% (35/85) | 74.5% (35/82) | 61.5% (72/164) | 60.9% (73/161) | not measured | not measured |
| umm-maybe Swin detector | 66.7% (25/7) | 48.4% (59/17) | 48.4% (60/16) | 58.3% (173/51) | 58.1% (175/48) | not measured | not measured |
| prithivMLmods SigLIP2 detector | 47.9% (15/8) | 57.3% (44/46) | 56.7% (43/46) | 48.2% (97/88) | 47.9% (96/88) | not measured | not measured |

## 3. Main test set by source

| Approach | Wikimedia Commons | Unsplash | Open Images | Commercial generators | Open-source generators |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 64/64 | 63/64 | 63/64 | 131/132 | 53/60 |
| Community Forensics detector | 64/64 | 62/64 | 64/64 | 95/132 | 59/60 |
| CNN of this project, setting 3 | 57/64 | 55/64 | 30/64 | 122/132 | 46/60 |
| CNN of this project, final (setting 4) | 54/64 | 54/64 | 31/64 | 122/132 | 46/60 |
| Ateeqq SigLIP detector | 50/64 | 51/64 | 49/64 | 109/132 | 27/60 |
| CNN of this project, first attempt (setting 1) | 40/64 | 43/64 | 29/64 | 125/132 | 47/60 |
| CNN of this project, setting 2 | 28/64 | 29/64 | 17/64 | 132/132 | 49/60 |
| Manish18 ViT detector | 25/64 | 24/64 | 23/64 | 120/132 | 44/60 |
| umm-maybe Swin detector | 59/64 | 64/64 | 50/64 | 31/132 | 20/60 |
| prithivMLmods SigLIP2 detector | 35/64 | 28/64 | 34/64 | 60/132 | 28/60 |

## 4. Main test set by generator and photo category

| Images | Claude | CommFor | CNN s3 | CNN final | Ateeqq | CNN first | CNN s2 | Manish18 | umm-maybe | prithiv |
|---|---|---|---|---|---|---|---|---|---|---|
| Real: Open Images | 63/64 | 64/64 | 30/64 | 31/64 | 49/64 | 29/64 | 17/64 | 23/64 | 50/64 | 34/64 |
| Real: Unsplash | 63/64 | 62/64 | 55/64 | 54/64 | 51/64 | 43/64 | 29/64 | 24/64 | 64/64 | 28/64 |
| Real: objects | 18/18 | 18/18 | 15/18 | 16/18 | 14/18 | 9/18 | 8/18 | 4/18 | 17/18 | 9/18 |
| Real: buildings | 10/10 | 10/10 | 9/10 | 6/10 | 9/10 | 4/10 | 2/10 | 5/10 | 9/10 | 4/10 |
| Real: landscapes | 10/10 | 10/10 | 9/10 | 10/10 | 10/10 | 10/10 | 9/10 | 6/10 | 10/10 | 5/10 |
| Real: people | 10/10 | 10/10 | 8/10 | 8/10 | 6/10 | 7/10 | 4/10 | 7/10 | 9/10 | 5/10 |
| Real: flowers | 8/8 | 8/8 | 8/8 | 8/8 | 6/8 | 5/8 | 1/8 | 1/8 | 8/8 | 7/8 |
| Real: food | 4/4 | 4/4 | 4/4 | 4/4 | 2/4 | 3/4 | 2/4 | 1/4 | 4/4 | 3/4 |
| Real: markets | 4/4 | 4/4 | 4/4 | 2/4 | 3/4 | 2/4 | 2/4 | 1/4 | 2/4 | 2/4 |
| AI: Dalle2 | 12/12 | 9/12 | 12/12 | 11/12 | 8/12 | 11/12 | 12/12 | 11/12 | 1/12 | 6/12 |
| AI: Dalle3 | 12/12 | 11/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 6/12 | 4/12 |
| AI: FLUX-dev | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 7/12 | 3/12 |
| AI: FLUX-schnell | 12/12 | 11/12 | 10/12 | 11/12 | 12/12 | 11/12 | 12/12 | 12/12 | 2/12 | 8/12 |
| AI: Firefly_Image2 | 12/12 | 7/12 | 7/12 | 8/12 | 5/12 | 9/12 | 12/12 | 9/12 | 1/12 | 6/12 |
| AI: Firefly_Image3 | 11/12 | 4/12 | 9/12 | 9/12 | 11/12 | 12/12 | 12/12 | 11/12 | 2/12 | 4/12 |
| AI: IdeogramV1 | 12/12 | 7/12 | 12/12 | 12/12 | 11/12 | 12/12 | 12/12 | 11/12 | 2/12 | 6/12 |
| AI: IdeogramV2 | 12/12 | 8/12 | 12/12 | 12/12 | 7/12 | 12/12 | 12/12 | 10/12 | 2/12 | 5/12 |
| AI: Imagen3 | 12/12 | 9/12 | 12/12 | 12/12 | 10/12 | 12/12 | 12/12 | 12/12 | 3/12 | 7/12 |
| AI: MidjourneyV5_2 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 10/12 | 2/12 | 8/12 |
| AI: MidjourneyV6_1 | 12/12 | 5/12 | 12/12 | 11/12 | 9/12 | 10/12 | 12/12 | 10/12 | 3/12 | 3/12 |
| AI: DFGAN | 5/6 | 6/6 | 3/6 | 3/6 | 1/6 | 3/6 | 1/6 | 3/6 | 6/6 | 5/6 |
| AI: DeciDiffusionV2 | 6/6 | 6/6 | 6/6 | 4/6 | 4/6 | 5/6 | 6/6 | 4/6 | 0/6 | 3/6 |
| AI: GALIP | 5/6 | 5/6 | 2/6 | 2/6 | 0/6 | 5/6 | 3/6 | 1/6 | 3/6 | 5/6 |
| AI: Hourglass | 1/6 | 6/6 | 4/6 | 5/6 | 2/6 | 4/6 | 4/6 | 2/6 | 4/6 | 1/6 |
| AI: LCM_lora_sdv15 | 6/6 | 6/6 | 6/6 | 6/6 | 4/6 | 6/6 | 6/6 | 6/6 | 1/6 | 2/6 |
| AI: LCM_lora_sdxl | 6/6 | 6/6 | 4/6 | 3/6 | 2/6 | 5/6 | 5/6 | 6/6 | 1/6 | 4/6 |
| AI: LCM_lora_ssd1b | 6/6 | 6/6 | 4/6 | 5/6 | 3/6 | 5/6 | 6/6 | 6/6 | 1/6 | 3/6 |
| AI: kandinsky_2_2 | 6/6 | 6/6 | 5/6 | 6/6 | 3/6 | 4/6 | 6/6 | 5/6 | 1/6 | 1/6 |
| AI: kvikontent_midjourney_v6 | 6/6 | 6/6 | 6/6 | 6/6 | 2/6 | 4/6 | 6/6 | 5/6 | 3/6 | 3/6 |
| AI: stable_cascade | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 | 0/6 | 1/6 |

## 5. Images from ChatGPT and Gemini

An extra test set from the test split of OpenFake (version 2.1): images made with GPT Image 2 and GPT Image 1.5, the image generators of ChatGPT, and with Nano Banana Pro, the image generator of Gemini, together with real photographs from two further sources. Cells: images classified correctly.

## 6. Sensitivity to image processing (check set)

Images of the check set (69 real, 88 AI) still classified correctly after one processing step, as real / AI. A detector that reads sharpness or cleanness instead of the way an image was made loses real photographs here.

| Approach | as downloaded | Gaussian blur, radius 0.6 | Gaussian blur, radius 1.0 | unsharp mask (sharpen) | downscaled to half size | Gaussian noise, sd 3 | Mean accuracy |
|---|---|---|---|---|---|---|---|
| CNN of this project, final (setting 4) | 59 / 81 | 40 / 81 | 45 / 78 | 62 / 80 | 41 / 86 | 63 / 78 | 84.3% |
| CNN of this project, setting 3 | 58 / 81 | 34 / 82 | 23 / 81 | 64 / 81 | 50 / 83 | 32 / 84 | 79.9% |
| CNN of this project, first attempt (setting 1) | 41 / 84 | 36 / 84 | 32 / 87 | 43 / 84 | 41 / 84 | 35 / 84 | 78.0% |
| CNN of this project, setting 2 | 33 / 87 | 15 / 87 | 2 / 88 | 43 / 87 | 34 / 87 | 27 / 88 | 72.0% |

## 7. Ranking quality, confidence and speed

AUC = probability that a real photograph gets a higher 'real' score than a generated image on the main test set (1.0 = perfect, 0.5 = guessing); it does not depend on the 50% decision threshold. Seconds = time per image on this laptop's processor, without a GPU.

| Approach | AUC | Correct when confidence is 90% or more | Correct when confidence is below 75% | Seconds per image (mean) | Seconds (slowest image) |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | no scores |  |  |  |  |
| Community Forensics detector | 0.972 | 94% of 342 | 57% of 21 | 0.329 | 0.466 |
| CNN of this project, setting 3 | 0.906 | 88% of 250 | 62% of 56 | 3.568 | 9.603 |
| CNN of this project, final (setting 4) | 0.895 | 85% of 248 | 61% of 62 | 3.071 | 9.533 |
| Ateeqq SigLIP detector | 0.797 | 75% of 374 | 50% of 8 | 0.326 | 0.457 |
| CNN of this project, first attempt (setting 1) | 0.887 | 80% of 302 | 46% of 41 | 0.229 | 0.355 |
| CNN of this project, setting 2 | 0.880 | 71% of 294 | 57% of 56 | 3.606 | 9.504 |
| Manish18 ViT detector | 0.753 | 65% of 297 | 46% of 41 | 0.292 | 0.344 |
| umm-maybe Swin detector | 0.637 | 63% of 150 | 54% of 125 | 0.401 | 0.505 |
| prithivMLmods SigLIP2 detector | 0.490 | 70% of 43 | 46% of 250 | 0.295 | 0.484 |

## 8. What has been measured

| Approach | Pilot (48) | Check (157) | Check, JPEG 90 | Main (384) | Main, JPEG 90 | ChatGPT and Gemini set (168) | ChatGPT and Gemini set, JPEG 90 |
|---|---|---|---|---|---|---|---|
| Claude Sonnet 5.5 | yes | not planned | not planned | yes | not planned | **missing** | not planned |
| Community Forensics detector | yes | yes | yes | yes | yes | **missing** | **missing** |
| CNN of this project, setting 3 | yes | yes | yes | yes | yes | **missing** | **missing** |
| CNN of this project, final (setting 4) | yes | yes | yes | yes | yes | **missing** | **missing** |
| Ateeqq SigLIP detector | yes | yes | yes | yes | yes | **missing** | **missing** |
| CNN of this project, first attempt (setting 1) | yes | yes | yes | yes | yes | **missing** | **missing** |
| CNN of this project, setting 2 | yes | yes | yes | yes | yes | **missing** | **missing** |
| Manish18 ViT detector | yes | yes | yes | yes | yes | **missing** | **missing** |
| umm-maybe Swin detector | yes | yes | yes | yes | yes | **missing** | **missing** |
| prithivMLmods SigLIP2 detector | yes | yes | yes | yes | yes | **missing** | **missing** |

19 combinations are missing. Claude Sonnet 5.5: ChatGPT and Gemini set (168); Community Forensics detector: ChatGPT and Gemini set (168); Community Forensics detector: ChatGPT and Gemini set, JPEG 90; CNN of this project, setting 3: ChatGPT and Gemini set (168); CNN of this project, setting 3: ChatGPT and Gemini set, JPEG 90; CNN of this project, final (setting 4): ChatGPT and Gemini set (168); CNN of this project, final (setting 4): ChatGPT and Gemini set, JPEG 90; Ateeqq SigLIP detector: ChatGPT and Gemini set (168); Ateeqq SigLIP detector: ChatGPT and Gemini set, JPEG 90; CNN of this project, first attempt (setting 1): ChatGPT and Gemini set (168); CNN of this project, first attempt (setting 1): ChatGPT and Gemini set, JPEG 90; CNN of this project, setting 2: ChatGPT and Gemini set (168); CNN of this project, setting 2: ChatGPT and Gemini set, JPEG 90; Manish18 ViT detector: ChatGPT and Gemini set (168); Manish18 ViT detector: ChatGPT and Gemini set, JPEG 90; umm-maybe Swin detector: ChatGPT and Gemini set (168); umm-maybe Swin detector: ChatGPT and Gemini set, JPEG 90; prithivMLmods SigLIP2 detector: ChatGPT and Gemini set (168); prithivMLmods SigLIP2 detector: ChatGPT and Gemini set, JPEG 90.
