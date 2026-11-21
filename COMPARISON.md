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
- **CNN of this project, third round (adopted candidate):** the candidate adopted by the rule of the third round, its threshold built in.
- **CNN of this project, third round, candidate A:** candidate A, decision threshold chosen on the 1,213 development images of the third round.
- **CNN of this project, third round, candidate A before the threshold:** setting 5 trained further on 4,920 new OpenFake images, 2 epochs, learning rate 1e-5; default threshold of 50%.
- **CNN of this project, third round, candidate B:** candidate B, decision threshold chosen on the 1,213 development images of the third round.
- **CNN of this project, third round, candidate B before the threshold:** as candidate A, with the processing of a sharing platform in the augmentation; default threshold of 50%.
- **CNN of this project, third round, candidate B, run with seed 1:** candidate B trained again with another seed, its threshold chosen on the development data and built in.
- **CNN of this project, third round, candidate B, run with seed 1, before the threshold:** candidate B trained again with another seed; default threshold of 50%.
- **CNN of this project, third round, candidate B, run with seed 2:** candidate B trained again with another seed, its threshold chosen on the development data and built in.
- **CNN of this project, third round, candidate B, run with seed 2, before the threshold:** candidate B trained again with another seed; default threshold of 50%.
- **CNN of this project, third round, candidate C:** candidate C, decision threshold chosen on the 1,213 development images of the third round.
- **CNN of this project, third round, candidate C before the threshold:** the base network trained on the 4,920 new OpenFake images only, 3 epochs, learning rate 5e-5; default threshold of 50%.
- **Combination of the second round: CNN of this project (setting 5) and Community Forensics detector:** weighted mean of the two detectors' scores; weight and threshold chosen on development data.
- **Combination of the third round, with the run of candidate B with seed 1:** as the combination of the third round, with the CNN of the run of seed 1; threshold chosen again on the development data.
- **Combination of the third round, with the run of candidate B with seed 2:** as the combination of the third round, with the CNN of the run of seed 2; threshold chosen again on the development data.
- **Combination of the third round: CNN of this project (candidate B) and Community Forensics detector:** weighted mean of the two detectors' scores; weight and threshold chosen on the development data of the third round.
- **Community Forensics detector, threshold tuned on the development images:** as the published detector; only its decision threshold was chosen on the development images of this project.
- **Community Forensics detector, threshold tuned on the development images of the third round:** as the published detector; only its decision threshold was chosen on the 1,213 development images of the third round.
- **Ateeqq SigLIP 2 detector:** card: 60,000 AI-generated and 60,000 human images; card reports 99.2% on its own test split.
- **Community Forensics detector:** the authors' full Community Forensics data: 2.7 million images from 4,803 generators.
- **Manish18 ViT detector:** not stated: the model card is an unfilled template.
- **prithivMLmods SigLIP 2 detector:** card: its own deepfake data; card reports 94.4% on its own test data.
- **umm-maybe Swin detector:** card: made in October 2022 for artistic images, 'not a deepfake photo detector'.
- **Claude Sonnet 5.5:** not a detector; general-purpose model of Anthropic.

## 2. Accuracy on every set

Percent of images classified correctly (real photographs correct / AI images correct). JPEG 90 = every image re-compressed first.

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
| Ateeqq SigLIP 2 detector | 77.3% (162/135) | 77.6% (161/137) | 81.0% (161/150) | 80.5% (156/153) | 82.0% (146/169) | 82.0% (145/170) | 91.7% (0/44) | 91.7% (0/44) |
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
| Ateeqq SigLIP 2 detector | 75.8% (160/131) | 75.8% (159/132) | 82.1% (61/77) | 81.0% (58/78) | 83.3% (50/50) | 83.3% (50/50) |
| Manish18 ViT detector | 66.4% (87/168) | 63.3% (87/156) | 70.8% (42/77) | 72.0% (45/76) | 57.5% (22/47) | 57.5% (22/47) |
| umm-maybe Swin detector | 59.4% (165/63) | 56.2% (164/52) | 52.4% (59/29) | 54.2% (63/28) | 59.2% (53/18) | 60.0% (54/18) |
| prithivMLmods SigLIP 2 detector | 50.0% (102/90) | 49.7% (101/90) | 57.7% (45/52) | 57.1% (43/53) | 46.7% (26/30) | 46.7% (26/30) |

**Third round: development data (the five sets scored in the first two rounds), models of the third round**

| Approach | Check set (157) | Check set (157), JPEG 90 | Development test set (384) | Development test set (384), JPEG 90 | Final test set (384) | Final test set (384), JPEG 90 | ChatGPT and Gemini set (168) | ChatGPT and Gemini set (168), JPEG 90 | Reddit set (120) | Reddit set (120), JPEG 90 |
|---|---|---|---|---|---|---|---|---|---|---|
| CNN of this project, third round (adopted candidate) | 89.8% (67/74) | 81.5% (67/61) | 80.5% (170/139) | 78.6% (174/128) | 81.8% (175/139) | 77.1% (178/118) | 81.0% (62/74) | 80.4% (67/68) | 83.3% (57/43) | 81.7% (57/41) |
| CNN of this project, third round, candidate B, run with seed 1 | 89.8% (66/75) | 84.7% (66/67) | 79.4% (164/141) | 76.8% (167/128) | 82.3% (173/143) | 78.4% (173/128) | 78.0% (58/73) | 80.4% (64/71) | 83.3% (56/44) | 81.7% (56/42) |
| CNN of this project, third round, candidate B, run with seed 2 | 89.2% (64/76) | 85.4% (66/68) | 80.2% (165/143) | 78.9% (171/132) | 82.0% (171/144) | 77.9% (171/128) | 78.0% (57/74) | 78.6% (62/70) | 83.3% (56/44) | 84.2% (57/44) |
| CNN of this project, setting 5 before the threshold | 87.3% (62/75) | 83.4% (63/68) | 81.0% (155/156) | 80.2% (161/147) | 82.8% (165/153) | 80.2% (169/139) | 64.9% (52/57) | 63.1% (57/49) | 68.3% (56/26) | 68.3% (56/26) |
| CNN of this project, third round, candidate A before the threshold | 87.3% (56/81) | 86.0% (59/76) | 76.6% (131/163) | 75.8% (138/153) | 77.3% (136/161) | 76.8% (142/153) | 75.0% (44/82) | 77.4% (50/80) | 85.8% (52/51) | 83.3% (51/49) |
| Community Forensics detector | 86.0% (69/66) | 75.2% (69/49) | 89.6% (190/154) | 81.0% (190/121) | 92.2% (190/164) | 82.3% (190/126) | 57.7% (71/26) | 50.6% (71/14) | 64.2% (60/17) | 65.8% (60/19) |
| CNN of this project, third round, candidate B before the threshold | 86.0% (53/82) | 84.7% (55/78) | 73.2% (116/165) | 72.9% (121/159) | 75.0% (123/165) | 74.7% (129/158) | 75.6% (43/84) | 76.8% (47/82) | 81.7% (45/53) | 80.8% (46/51) |
| CNN of this project, third round, candidate B, run with seed 1, before the threshold | 84.1% (51/81) | 84.7% (55/78) | 74.7% (126/161) | 76.6% (134/160) | 75.5% (127/163) | 76.8% (137/158) | 76.2% (43/85) | 75.0% (46/80) | 85.0% (50/52) | 84.2% (51/50) |
| CNN of this project, third round, candidate B, run with seed 2, before the threshold | 84.1% (50/82) | 83.4% (52/79) | 74.2% (121/164) | 73.7% (125/158) | 76.6% (128/166) | 75.8% (133/158) | 74.4% (41/84) | 74.4% (45/80) | 82.5% (47/52) | 83.3% (49/51) |
| CNN of this project, third round, candidate C before the threshold | 73.2% (39/76) | 70.7% (39/72) | 68.5% (114/149) | 67.4% (111/148) | 68.8% (124/140) | 67.2% (126/132) | 83.9% (49/92) | 82.7% (50/89) | 87.5% (49/56) | 85.0% (48/54) |

**First and second rounds: development data**

| Approach | Pilot test set (48) | Check set (157) | Check set (157), JPEG 90 | Development test set (384) | Development test set (384), JPEG 90 |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 97.9% (26/21) |  |  | 97.4% (190/184) |  |
| CNN of this project, setting 3 | 97.9% (25/22) | 88.5% (58/81) | 87.9% (60/78) | 80.7% (142/168) | 80.2% (151/157) |
| Community Forensics detector | 95.8% (26/20) | 86.0% (69/66) | 75.2% (69/49) | 89.6% (190/154) | 81.0% (190/121) |
| Ateeqq SigLIP 2 detector | 87.5% (24/18) | 79.6% (56/69) | 79.6% (56/69) | 74.5% (150/136) | 74.7% (147/140) |
| CNN of this project, setting 4 (first round) | 87.5% (21/21) | 89.2% (59/81) | 87.3% (60/77) | 79.9% (139/168) | 81.8% (150/164) |
| Manish18 ViT detector | 72.9% (16/19) | 76.4% (35/85) | 74.5% (35/82) | 61.5% (72/164) | 60.9% (73/161) |
| CNN of this project, first attempt (setting 1) | 70.8% (14/20) | 79.6% (41/84) | 79.6% (42/83) | 74.0% (112/172) | 73.4% (121/161) |
| umm-maybe Swin detector | 66.7% (25/7) | 48.4% (59/17) | 48.4% (60/16) | 58.3% (173/51) | 58.1% (175/48) |
| CNN of this project, setting 2 | 64.6% (10/21) | 76.4% (33/87) | 77.1% (35/86) | 66.4% (74/181) | 66.9% (78/179) |
| prithivMLmods SigLIP 2 detector | 47.9% (15/8) | 57.3% (44/46) | 56.7% (43/46) | 48.2% (97/88) | 47.9% (96/88) |
| Combination of the second round: CNN of this project (setting 5) and Community Forensics detector |  | 92.4% (69/76) | 86.6% (69/67) | 93.8% (184/176) | 89.8% (186/159) |
| CNN of this project, setting 5 (published) |  | 88.5% (66/73) | 83.4% (66/65) | 82.6% (167/150) | 80.2% (170/138) |
| CNN of this project, setting 5, second run |  | 87.3% (64/73) | 83.4% (64/67) | 84.1% (167/156) | 81.5% (170/143) |
| CNN of this project, setting 5, third run |  | 90.4% (66/76) | 86.0% (66/69) | 85.4% (168/160) | 82.0% (171/144) |
| CNN of this project, setting 5 before the threshold |  | 87.3% (62/75) | 83.4% (63/68) | 81.0% (155/156) | 80.2% (161/147) |
| CNN of this project, setting 5, second run, before the threshold |  | 87.3% (65/72) | 82.8% (66/64) | 83.6% (170/151) | 82.3% (177/139) |
| CNN of this project, setting 5, third run, before the threshold |  | 87.9% (67/71) | 83.4% (67/64) | 83.3% (173/147) | 79.4% (177/128) |

## 3. Final test set 3: images classified correctly, by source and generator

| Approach | Open Images (64) | Unsplash (64) | Wikimedia Commons (64) | Commercial generators (132) | Open-source generators (60) |
|---|---|---|---|---|---|
| Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | 64 | 63 | 62 | 114 | 60 |
| Combination of the third round, with the run of candidate B with seed 1 | 61 | 61 | 59 | 119 | 60 |
| Combination of the third round, with the run of candidate B with seed 2 | 60 | 62 | 59 | 117 | 60 |
| Combination of the third round: CNN of this project (candidate B) and Community Forensics detector | 59 | 60 | 57 | 119 | 60 |
| Community Forensics detector, threshold tuned on the development images of the third round | 60 | 56 | 61 | 116 | 60 |
| Community Forensics detector | 64 | 64 | 62 | 91 | 60 |
| CNN of this project, setting 5 (published) | 47 | 63 | 62 | 109 | 37 |
| CNN of this project, third round (adopted candidate) | 58 | 61 | 59 | 106 | 33 |
| CNN of this project, third round, candidate B | 58 | 61 | 59 | 106 | 33 |
| CNN of this project, third round, candidate A | 57 | 60 | 59 | 106 | 33 |
| CNN of this project, setting 4 (first round) | 31 | 56 | 55 | 124 | 48 |
| CNN of this project, third round, candidate B, run with seed 1 | 55 | 59 | 59 | 106 | 34 |
| CNN of this project, third round, candidate B, run with seed 2 | 55 | 58 | 57 | 108 | 34 |
| Ateeqq SigLIP 2 detector | 47 | 56 | 59 | 104 | 31 |
| CNN of this project, third round, candidate A before the threshold | 45 | 45 | 50 | 117 | 40 |
| CNN of this project, third round, candidate B before the threshold | 43 | 41 | 49 | 123 | 39 |
| CNN of this project, third round, candidate C | 52 | 51 | 58 | 94 | 28 |
| CNN of this project, third round, candidate C before the threshold | 24 | 41 | 44 | 112 | 35 |
| Manish18 ViT detector | 27 | 22 | 31 | 123 | 49 |
| umm-maybe Swin detector | 47 | 61 | 56 | 34 | 12 |
| prithivMLmods SigLIP 2 detector | 31 | 31 | 39 | 64 | 39 |

| Images | Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | Combination of the third round, with the run of candidate B with seed 1 | Combination of the third round, with the run of candidate B with seed 2 | Combination of the third round: CNN of this project (candidate B) and Community Forensics detector | Community Forensics detector, threshold tuned on the development images of the third round | Community Forensics detector | CNN of this project, setting 5 (published) | CNN of this project, third round (adopted candidate) | CNN of this project, third round, candidate B | CNN of this project, third round, candidate A | CNN of this project, setting 4 (first round) | CNN of this project, third round, candidate B, run with seed 1 | CNN of this project, third round, candidate B, run with seed 2 | Ateeqq SigLIP 2 detector | CNN of this project, third round, candidate A before the threshold | CNN of this project, third round, candidate B before the threshold | CNN of this project, third round, candidate C | CNN of this project, third round, candidate C before the threshold | Manish18 ViT detector | umm-maybe Swin detector | prithivMLmods SigLIP 2 detector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real: Open Images (64) | 64 | 61 | 60 | 59 | 60 | 64 | 47 | 58 | 58 | 57 | 31 | 55 | 55 | 47 | 45 | 43 | 52 | 24 | 27 | 47 | 31 |
| Real: Unsplash (64) | 63 | 61 | 62 | 60 | 56 | 64 | 63 | 61 | 61 | 60 | 56 | 59 | 58 | 56 | 45 | 41 | 51 | 41 | 22 | 61 | 31 |
| Real: plants (12) | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 10 | 12 | 12 | 10 | 11 | 11 | 12 | 8 | 1 | 12 | 7 |
| Real: horses (8) | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 7 | 6 | 5 | 4 | 6 |
| Real: lakes (8) | 8 | 6 | 6 | 6 | 8 | 8 | 7 | 6 | 6 | 6 | 6 | 6 | 6 | 7 | 5 | 4 | 7 | 7 | 5 | 7 | 6 |
| Real: statues (8) | 6 | 5 | 5 | 5 | 6 | 6 | 7 | 7 | 7 | 7 | 6 | 7 | 6 | 6 | 5 | 5 | 5 | 2 | 4 | 8 | 7 |
| Real: towers (8) | 8 | 8 | 8 | 7 | 7 | 8 | 8 | 7 | 7 | 7 | 6 | 7 | 7 | 8 | 5 | 5 | 8 | 5 | 3 | 8 | 2 |
| Real: waterfalls (8) | 8 | 8 | 8 | 7 | 8 | 8 | 8 | 7 | 7 | 7 | 8 | 7 | 7 | 8 | 7 | 7 | 8 | 8 | 6 | 6 | 5 |
| Real: boats (7) | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 7 | 5 | 5 | 7 | 5 | 4 | 7 | 5 |
| Real: palaces (4) | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 4 | 3 | 4 | 4 | 4 | 4 | 4 | 3 | 2 | 2 | 3 | 1 |
| Real: gardens (1) | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 0 | 0 | 1 | 1 | 1 | 1 | 0 |
| AI: Dalle2 (12) | 11 | 10 | 10 | 10 | 12 | 10 | 4 | 2 | 2 | 2 | 12 | 1 | 2 | 5 | 6 | 7 | 4 | 5 | 10 | 3 | 5 |
| AI: Dalle3 (12) | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 4 | 4 |
| AI: FLUX-dev (12) | 12 | 12 | 12 | 12 | 12 | 11 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 11 | 12 | 12 | 6 | 6 |
| AI: FLUX-schnell (12) | 12 | 12 | 12 | 12 | 12 | 11 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 10 | 11 | 12 | 5 | 5 |
| AI: Firefly_Image2 (12) | 8 | 9 | 8 | 9 | 8 | 7 | 6 | 6 | 6 | 6 | 8 | 6 | 7 | 9 | 8 | 10 | 6 | 8 | 12 | 1 | 5 |
| AI: Firefly_Image3 (12) | 4 | 6 | 5 | 6 | 7 | 1 | 7 | 6 | 6 | 6 | 8 | 6 | 6 | 8 | 8 | 12 | 5 | 8 | 10 | 1 | 7 |
| AI: IdeogramV1 (12) | 11 | 11 | 11 | 11 | 10 | 8 | 11 | 11 | 11 | 11 | 12 | 11 | 11 | 10 | 12 | 11 | 10 | 11 | 10 | 3 | 7 |
| AI: IdeogramV2 (12) | 12 | 12 | 12 | 12 | 12 | 9 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 7 | 12 | 12 | 8 | 11 | 10 | 5 | 8 |
| AI: Imagen3 (12) | 9 | 11 | 11 | 11 | 9 | 6 | 10 | 11 | 11 | 11 | 12 | 11 | 11 | 8 | 11 | 11 | 6 | 10 | 12 | 1 | 5 |
| AI: MidjourneyV5_2 (12) | 12 | 12 | 12 | 12 | 12 | 10 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 12 | 11 | 4 | 7 |
| AI: MidjourneyV6_1 (12) | 11 | 12 | 12 | 12 | 10 | 6 | 11 | 10 | 10 | 10 | 12 | 11 | 11 | 9 | 12 | 12 | 10 | 12 | 12 | 1 | 5 |
| AI: DFGAN (6) | 6 | 6 | 6 | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 | 2 | 1 | 1 | 1 | 3 | 1 | 5 |
| AI: DeciDiffusionV2 (6) | 6 | 6 | 6 | 6 | 6 | 6 | 5 | 5 | 5 | 5 | 6 | 6 | 6 | 3 | 6 | 6 | 4 | 4 | 6 | 2 | 4 |
| AI: GALIP (6) | 6 | 6 | 6 | 6 | 6 | 6 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 1 | 5 |
| AI: Hourglass (6) | 6 | 6 | 6 | 6 | 6 | 6 | 3 | 2 | 2 | 2 | 5 | 2 | 2 | 1 | 4 | 4 | 1 | 4 | 3 | 1 | 4 |
| AI: LCM_lora_sdv15 (6) | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 4 | 6 | 6 | 4 | 6 | 6 | 0 | 4 |
| AI: LCM_lora_sdxl (6) | 6 | 6 | 6 | 6 | 6 | 6 | 1 | 1 | 1 | 1 | 3 | 1 | 1 | 2 | 1 | 1 | 1 | 2 | 6 | 1 | 4 |
| AI: LCM_lora_ssd1b (6) | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 2 | 2 | 2 | 6 | 2 | 2 | 4 | 4 | 4 | 3 | 4 | 6 | 2 | 5 |
| AI: kandinsky_2_2 (6) | 6 | 6 | 6 | 6 | 6 | 6 | 5 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 5 | 5 | 6 | 1 | 2 |
| AI: kvikontent_midjourney_v6 (6) | 6 | 6 | 6 | 6 | 6 | 6 | 5 | 5 | 5 | 5 | 6 | 5 | 5 | 5 | 5 | 5 | 4 | 4 | 6 | 3 | 3 |
| AI: stable_cascade (6) | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 6 | 5 | 6 | 6 | 5 | 5 | 6 | 0 | 3 |

## 4. ChatGPT and Gemini set 3: images classified correctly, by source and generator

| Approach | DOCCI (96) | ImageNet (96) | Commercial generators (192) |
|---|---|---|---|
| CNN of this project, third round, candidate C | 90 | 86 | 166 |
| CNN of this project, third round, candidate C before the threshold | 74 | 62 | 183 |
| Combination of the third round: CNN of this project (candidate B) and Community Forensics detector | 94 | 76 | 144 |
| Ateeqq SigLIP 2 detector | 78 | 83 | 150 |
| Combination of the third round, with the run of candidate B with seed 2 | 94 | 81 | 134 |
| Combination of the third round, with the run of candidate B with seed 1 | 93 | 77 | 136 |
| CNN of this project, third round (adopted candidate) | 93 | 71 | 134 |
| CNN of this project, third round, candidate B | 93 | 71 | 134 |
| CNN of this project, third round, candidate B, run with seed 2 | 86 | 67 | 141 |
| CNN of this project, third round, candidate A | 92 | 64 | 138 |
| CNN of this project, third round, candidate B, run with seed 1 | 88 | 65 | 140 |
| Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | 95 | 96 | 99 |
| Community Forensics detector, threshold tuned on the development images of the third round | 94 | 95 | 94 |
| CNN of this project, third round, candidate A before the threshold | 80 | 39 | 159 |
| CNN of this project, setting 5 (published) | 90 | 65 | 121 |
| CNN of this project, third round, candidate B before the threshold | 70 | 39 | 162 |
| Manish18 ViT detector | 50 | 54 | 161 |
| CNN of this project, setting 4 (first round) | 89 | 20 | 150 |
| Community Forensics detector | 96 | 96 | 53 |
| prithivMLmods SigLIP 2 detector | 37 | 77 | 120 |
| umm-maybe Swin detector | 93 | 68 | 68 |

| Images | CNN of this project, third round, candidate C | CNN of this project, third round, candidate C before the threshold | Combination of the third round: CNN of this project (candidate B) and Community Forensics detector | Ateeqq SigLIP 2 detector | Combination of the third round, with the run of candidate B with seed 2 | Combination of the third round, with the run of candidate B with seed 1 | CNN of this project, third round (adopted candidate) | CNN of this project, third round, candidate B | CNN of this project, third round, candidate B, run with seed 2 | CNN of this project, third round, candidate A | CNN of this project, third round, candidate B, run with seed 1 | Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | Community Forensics detector, threshold tuned on the development images of the third round | CNN of this project, third round, candidate A before the threshold | CNN of this project, setting 5 (published) | CNN of this project, third round, candidate B before the threshold | Manish18 ViT detector | CNN of this project, setting 4 (first round) | Community Forensics detector | prithivMLmods SigLIP 2 detector | umm-maybe Swin detector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real: DOCCI (96) | 90 | 74 | 94 | 78 | 94 | 93 | 93 | 93 | 86 | 92 | 88 | 95 | 94 | 80 | 90 | 70 | 50 | 89 | 96 | 37 | 93 |
| Real: ImageNet (96) | 86 | 62 | 76 | 83 | 81 | 77 | 71 | 71 | 67 | 64 | 65 | 96 | 95 | 39 | 65 | 39 | 54 | 20 | 96 | 77 | 68 |
| AI: gpt-image-1.5 (48) | 42 | 45 | 39 | 37 | 35 | 37 | 35 | 35 | 36 | 36 | 36 | 23 | 26 | 40 | 24 | 41 | 46 | 36 | 13 | 18 | 26 |
| AI: gpt-image-2 (48) | 38 | 45 | 27 | 34 | 25 | 25 | 23 | 23 | 27 | 26 | 26 | 13 | 13 | 37 | 23 | 39 | 37 | 29 | 3 | 29 | 22 |
| AI: nano-banana (48) | 44 | 47 | 48 | 37 | 48 | 48 | 47 | 47 | 47 | 47 | 47 | 44 | 41 | 48 | 48 | 48 | 47 | 48 | 28 | 30 | 6 |
| AI: nano-banana-pro (48) | 42 | 46 | 30 | 42 | 26 | 26 | 29 | 29 | 31 | 29 | 31 | 19 | 14 | 34 | 26 | 34 | 31 | 37 | 9 | 43 | 14 |

## 5. Reddit set 3: images classified correctly, by source and generator

| Images | CNN of this project, third round, candidate C | CNN of this project, third round, candidate A before the threshold | CNN of this project, third round, candidate B, run with seed 1 | Combination of the third round: CNN of this project (candidate B) and Community Forensics detector | CNN of this project, third round, candidate B, run with seed 2 | CNN of this project, setting 4 (first round) | CNN of this project, third round (adopted candidate) | CNN of this project, third round, candidate B | CNN of this project, third round, candidate C before the threshold | CNN of this project, third round, candidate A | Combination of the third round, with the run of candidate B with seed 1 | Ateeqq SigLIP 2 detector | Combination of the third round, with the run of candidate B with seed 2 | CNN of this project, third round, candidate B before the threshold | CNN of this project, setting 5 (published) | Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | Community Forensics detector, threshold tuned on the development images of the third round | umm-maybe Swin detector | Community Forensics detector | Manish18 ViT detector | prithivMLmods SigLIP 2 detector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real: Reddit, photography subreddits (192) | 172 | 162 | 182 | 181 | 182 | 177 | 183 | 183 | 144 | 183 | 183 | 146 | 184 | 140 | 184 | 189 | 174 | 167 | 190 | 88 | 76 |
| AI: Reddit, AI subreddits (192) | 158 | 164 | 141 | 142 | 140 | 142 | 136 | 136 | 175 | 135 | 134 | 169 | 131 | 171 | 97 | 87 | 91 | 77 | 48 | 149 | 99 |

## 6. Seedream 5.0 control: images classified correctly, by source and generator

| Images | CNN of this project, third round, candidate B before the threshold | Ateeqq SigLIP 2 detector | CNN of this project, third round, candidate A before the threshold | Combination of the third round: CNN of this project (candidate B) and Community Forensics detector | CNN of this project, third round, candidate B, run with seed 2 | CNN of this project, third round, candidate B, run with seed 1 | Combination of the third round, with the run of candidate B with seed 1 | CNN of this project, third round (adopted candidate) | CNN of this project, third round, candidate B | CNN of this project, third round, candidate C before the threshold | CNN of this project, third round, candidate A | Combination of the third round, with the run of candidate B with seed 2 | Manish18 ViT detector | CNN of this project, third round, candidate C | Community Forensics detector, threshold tuned on the development images of the third round | CNN of this project, setting 4 (first round) | Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | CNN of this project, setting 5 (published) | prithivMLmods SigLIP 2 detector | Community Forensics detector | umm-maybe Swin detector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AI: seedream-v5.0 (48) | 46 | 44 | 43 | 41 | 40 | 39 | 39 | 38 | 38 | 38 | 37 | 36 | 33 | 29 | 24 | 23 | 21 | 19 | 19 | 15 | 6 |

## 7. Final test set: images classified correctly, by source and generator

| Approach | Open Images (64) | Unsplash (64) | Wikimedia Commons (64) | Commercial generators (132) | Open-source generators (60) |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 64 | 62 | 64 | 131 | 50 |
| Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | 62 | 62 | 64 | 118 | 60 |
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

| Images | Claude Sonnet 5.5 | Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | Community Forensics detector, threshold tuned on the development images | Community Forensics detector | CNN of this project, setting 5, third run | CNN of this project, setting 5, second run | CNN of this project, setting 4, threshold tuned on the development images | CNN of this project, setting 5 before the threshold | CNN of this project, setting 4 (first round) | CNN of this project, setting 5 (published) | CNN of this project, first attempt (setting 1) | Ateeqq SigLIP 2 detector | Manish18 ViT detector | umm-maybe Swin detector | prithivMLmods SigLIP 2 detector |
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

## 8. ChatGPT and Gemini set: images classified correctly, by source and generator

| Approach | DOCCI (36) | ImageNet (36) | Commercial generators (96) |
|---|---|---|---|
| Claude Sonnet 5.5 | 31 | 31 | 81 |
| Ateeqq SigLIP 2 detector | 28 | 33 | 77 |
| CNN of this project, setting 4 (first round) | 34 | 9 | 80 |
| CNN of this project, setting 5, third run | 34 | 26 | 59 |
| Manish18 ViT detector | 23 | 19 | 77 |
| CNN of this project, setting 4, threshold tuned on the development images | 36 | 11 | 70 |
| Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | 35 | 36 | 45 |
| CNN of this project, setting 5 (published) | 35 | 26 | 54 |
| CNN of this project, setting 5, second run | 35 | 17 | 62 |
| CNN of this project, setting 5 before the threshold | 35 | 17 | 57 |
| Community Forensics detector, threshold tuned on the development images | 34 | 36 | 33 |
| CNN of this project, first attempt (setting 1) | 27 | 14 | 59 |
| Community Forensics detector | 35 | 36 | 26 |
| prithivMLmods SigLIP 2 detector | 15 | 30 | 52 |
| umm-maybe Swin detector | 32 | 27 | 29 |

| Images | Claude Sonnet 5.5 | Ateeqq SigLIP 2 detector | CNN of this project, setting 4 (first round) | CNN of this project, setting 5, third run | Manish18 ViT detector | CNN of this project, setting 4, threshold tuned on the development images | Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | CNN of this project, setting 5 (published) | CNN of this project, setting 5, second run | CNN of this project, setting 5 before the threshold | Community Forensics detector, threshold tuned on the development images | CNN of this project, first attempt (setting 1) | Community Forensics detector | prithivMLmods SigLIP 2 detector | umm-maybe Swin detector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real: DOCCI (36) | 31 | 28 | 34 | 34 | 23 | 36 | 35 | 35 | 35 | 35 | 34 | 27 | 35 | 15 | 32 |
| Real: ImageNet (36) | 31 | 33 | 9 | 26 | 19 | 11 | 36 | 26 | 17 | 17 | 36 | 14 | 36 | 30 | 27 |
| AI: gpt-image-1.5 (24) | 19 | 17 | 21 | 11 | 20 | 16 | 13 | 10 | 11 | 11 | 11 | 15 | 8 | 11 | 11 |
| AI: gpt-image-2 (24) | 21 | 17 | 18 | 14 | 19 | 16 | 5 | 12 | 16 | 13 | 2 | 14 | 2 | 8 | 6 |
| AI: nano-banana (24) | 21 | 21 | 24 | 22 | 23 | 23 | 21 | 23 | 23 | 24 | 17 | 16 | 13 | 18 | 3 |
| AI: nano-banana-pro (24) | 20 | 22 | 17 | 12 | 15 | 15 | 6 | 9 | 12 | 9 | 3 | 14 | 3 | 15 | 9 |

## 9. Reddit set: images classified correctly, by source and generator

| Images | Claude Sonnet 5.5 | Ateeqq SigLIP 2 detector | CNN of this project, setting 4 (first round) | CNN of this project, setting 4, threshold tuned on the development images | CNN of this project, setting 5, third run | Community Forensics detector, threshold tuned on the development images | Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | CNN of this project, first attempt (setting 1) | CNN of this project, setting 5, second run | CNN of this project, setting 5 (published) | CNN of this project, setting 5 before the threshold | Community Forensics detector | umm-maybe Swin detector | Manish18 ViT detector | prithivMLmods SigLIP 2 detector |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real: Reddit, photography subreddits (60) | 59 | 50 | 55 | 56 | 56 | 58 | 59 | 38 | 57 | 56 | 56 | 60 | 53 | 22 | 26 |
| AI: Reddit, AI subreddits (60) | 57 | 50 | 40 | 30 | 29 | 27 | 26 | 46 | 27 | 26 | 26 | 17 | 18 | 47 | 30 |

## 10. Development test set: images classified correctly, by source and generator

| Approach | Open Images (64) | Unsplash (64) | Wikimedia Commons (64) | Commercial generators (132) | Open-source generators (60) |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 63 | 63 | 64 | 131 | 53 |
| Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | 61 | 61 | 62 | 117 | 59 |
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

| Images | Claude Sonnet 5.5 | Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | Community Forensics detector | CNN of this project, setting 5, third run | CNN of this project, setting 5, second run | CNN of this project, setting 5, second run, before the threshold | CNN of this project, setting 5, third run, before the threshold | CNN of this project, setting 5 (published) | CNN of this project, setting 5 before the threshold | CNN of this project, setting 3 | CNN of this project, setting 4 (first round) | Ateeqq SigLIP 2 detector | CNN of this project, first attempt (setting 1) | CNN of this project, setting 2 | Manish18 ViT detector | umm-maybe Swin detector | prithivMLmods SigLIP 2 detector |
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
| CNN of this project, third round, candidate A before the threshold | 56 / 81 | 42 / 81 | 53 / 79 | 59 / 81 | 58 / 82 | 59 / 80 | 86.1% |
| CNN of this project, setting 5 before the threshold | 62 / 75 | 53 / 77 | 56 / 75 | 63 / 75 | 61 / 77 | 63 / 73 | 86.0% |
| CNN of this project, third round, candidate B before the threshold | 53 / 82 | 36 / 82 | 44 / 82 | 58 / 82 | 56 / 83 | 57 / 81 | 84.5% |
| CNN of this project, setting 4 (first round) | 59 / 81 | 40 / 81 | 45 / 78 | 62 / 80 | 41 / 86 | 63 / 78 | 84.3% |
| Community Forensics detector | 69 / 66 | 69 / 65 | 69 / 65 | 69 / 62 | 69 / 64 | 64 / 52 | 83.1% |
| CNN of this project, setting 3 | 58 / 81 | 34 / 82 | 23 / 81 | 64 / 81 | 50 / 83 | 32 / 84 | 79.9% |
| Ateeqq SigLIP 2 detector | 56 / 69 | 56 / 69 | 56 / 70 | 57 / 69 | 56 / 69 | 56 / 69 | 79.8% |
| CNN of this project, first attempt (setting 1) | 41 / 84 | 36 / 84 | 32 / 87 | 43 / 84 | 41 / 84 | 35 / 84 | 78.0% |
| Manish18 ViT detector | 35 / 85 | 35 / 85 | 35 / 84 | 35 / 84 | 35 / 85 | 35 / 83 | 76.0% |
| CNN of this project, third round, candidate C before the threshold | 39 / 76 | 39 / 71 | 38 / 78 | 43 / 74 | 32 / 80 | 49 / 68 | 72.9% |
| CNN of this project, setting 2 | 33 / 87 | 15 / 87 | 2 / 88 | 43 / 87 | 34 / 87 | 27 / 88 | 72.0% |
| prithivMLmods SigLIP 2 detector | 44 / 46 | 44 / 46 | 43 / 47 | 44 / 46 | 44 / 46 | 44 / 46 | 57.3% |
| umm-maybe Swin detector | 59 / 17 | 59 / 17 | 59 / 20 | 58 / 15 | 58 / 19 | 56 / 25 | 49.0% |

## Ranking quality, confidence and speed (final test set 3)

AUC = probability that a real photograph gets a higher 'real' score than a generated image (1.0 = perfect, 0.5 = guessing); it does not depend on the decision threshold. Seconds = time per image on this laptop's processor, without a GPU.

| Approach | AUC | Correct when the score is 0.9 or more | Correct when the score is below 0.75 | Seconds per image (mean) | Seconds (slowest image) |
|---|---|---|---|---|---|
| Combination of the second round: CNN of this project (setting 5) and Community Forensics detector | 0.980 | 98% of 263 | 84% of 62 |  |  |
| Combination of the third round, with the run of candidate B with seed 1 | 0.981 | 100% of 183 | 85% of 109 |  |  |
| Combination of the third round, with the run of candidate B with seed 2 | 0.981 | 100% of 198 | 83% of 102 |  |  |
| Combination of the third round: CNN of this project (candidate B) and Community Forensics detector | 0.981 | 100% of 191 | 80% of 111 |  |  |
| Community Forensics detector, threshold tuned on the development images of the third round | 0.973 | 95% of 332 | 69% of 45 |  |  |
| Community Forensics detector | 0.973 | 93% of 348 | 44% of 16 | 0.202 | 0.238 |
| CNN of this project, setting 5 (published) | 0.902 | 90% of 280 | 53% of 45 | 2.091 | 5.602 |
| CNN of this project, third round (adopted candidate) | 0.878 | 93% of 194 | 65% of 81 |  |  |
| CNN of this project, third round, candidate B | 0.878 | 94% of 152 | 72% of 152 |  |  |
| CNN of this project, third round, candidate A | 0.878 | 94% of 162 | 69% of 132 |  |  |
| CNN of this project, setting 4 (first round) | 0.920 | 90% of 249 | 57% of 74 | 2.058 | 5.417 |
| CNN of this project, third round, candidate B, run with seed 1 | 0.877 | 94% of 181 | 60% of 94 |  |  |
| CNN of this project, third round, candidate B, run with seed 2 | 0.881 | 94% of 188 | 57% of 90 |  |  |
| Ateeqq SigLIP 2 detector | 0.842 | 78% of 375 | 50% of 4 | 0.301 | 0.38 |
| CNN of this project, third round, candidate A before the threshold | 0.878 | 94% of 162 | 55% of 132 |  |  |
| CNN of this project, third round, candidate B before the threshold | 0.878 | 94% of 152 | 61% of 142 |  |  |
| CNN of this project, third round, candidate C | 0.780 | 97% of 36 | 68% of 206 |  |  |
| CNN of this project, third round, candidate C before the threshold | 0.780 | 97% of 36 | 55% of 206 |  |  |
| Manish18 ViT detector | 0.841 | 72% of 298 | 38% of 37 | 0.247 | 0.301 |
| umm-maybe Swin detector | 0.613 | 61% of 131 | 50% of 134 | 0.288 | 0.399 |
| prithivMLmods SigLIP 2 detector | 0.541 | 62% of 26 | 51% of 276 | 0.299 | 0.369 |

## What the comparison shows

Notes on the tables above. Every number in them is filled in from the results database. Statistical tests are exact McNemar tests on the same images; no correction for multiple comparisons was applied, and only the primary comparisons of each round were fixed in advance in [PROTOCOL.md](PROTOCOL.md).

<!-- live-system: the one sentence that names the system of the public demonstration; change it when that changes -->
**Status at the time of writing (21 November 2026):** the public demonstration (the page on Amazon Web Services and the model and demo page on Hugging Face) still runs the system of the second round; whether the system of the third round replaces it has not been decided.

### Third round

The five sets scored in the first two rounds became the development data (1,213 images: 925 of old kinds in the check set, the development test set and the final test set; 288 of new kinds in the ChatGPT and Gemini set and the Reddit set). Three candidates were trained on 4,920 new images from OpenFake v2.1 only (2,438 photographs, 2,482 generated images). By the rule committed in advance candidate B was adopted (setting 5 trained further with the processing of a sharing platform in the augmentation) and combined with the Community Forensics detector at weight 0.5 and threshold 0.2804. The new held-out sets are final test set 3 (384 images), ChatGPT and Gemini set 3 (384), Reddit set 3 (384) and a Seedream 5.0 control of 48 generated images.

1. **Primary comparisons: the combination of the third round against that of the second.** On ChatGPT and Gemini set 3 and Reddit set 3 together (768 images) 82.9% against 73.7% (102 images right only with the new system, 31 only with the old; p = 4.9e-10); after JPEG 90 82.2% against 70.3% (p = 1.9e-15). On final test set 3 92.4% against 94.5% (6 / 14; p = 0.12, not significant); after JPEG 90 90.6% against 89.8%. The gain is on the new kinds of image, not overall.
2. **The cost: more false alarms on photographs.** The new combination accepts 176 of the 192 photographs of final test set 3 (second round: 189) and 76 of the 96 ImageNet photographs of ChatGPT and Gemini set 3 (second round: 96); it recognises more generated images (179 against 174 on final test set 3, 144 against 99 and 142 against 87 on the sets of new kinds). In the processing test of the check set (default threshold) candidate B accepted only 36 of 69 photographs after a slight blur (radius 0.6).
3. **The network alone.** Candidate B with its threshold against setting 5 (published): final test set 3 82.6% against 82.8% (p = 1.00), ChatGPT and Gemini set 3 77.6% against 71.9% (p = 0.006), Reddit set 3 83.1% against 73.2% (p = 7.0e-08).
4. **Forgetting.** Candidate C, trained from the base network on the new data only, was rejected on the development data because it lost the older generators (it recognised 36 of the 120 development images of open-source generators, setting 5 74). The held-out sets confirm it: C is the most accurate approach on ChatGPT and Gemini set 3 (89.1%) and Reddit set 3 (85.9%) and the least accurate network on final test set 3 (73.7%).
5. **Variation between runs.** Candidate B was trained three times (seeds 42, 1, 2). The combination scores 82.9%, 81.1% and 81.2% on the sets of new kinds and 92.4%, 93.8% and 93.2% on final test set 3: the published seed-42 run is the best of the three on the new kinds and the lowest on final test set 3. The gain on the new kinds holds for all three runs; the difference on final test set 3 is not significant for any. The Seedream 5.0 control varies most (75.0% to 85.4% for the combination), and it has only 48 images.
6. **Second opinion.** The new combination and the Ateeqq detector agree on 74.5% of final test set 3, 72.7% of ChatGPT and Gemini set 3 and 74.0% of Reddit set 3; where they agree the answer is right for 96.9%, 93.2% and 94.7% of the images.
7. **Limits of the third round.** The generators of ChatGPT and Gemini set 3 and the subreddits of Reddit set 3 are in the new training data (from the same platforms, at other times), so these sets measure new images of the kinds trained on and may flatter the round; final test set 3 and the Seedream control measure generators absent from all training data. Labels of the Reddit images follow the subreddit. Deviations are recorded in PROTOCOL.md: two of the eight AI subreddits are not represented in the training data, one image of ChatGPT and Gemini set 3 is a variant of one in the development data, the published detectors were read offline from the local copies used in the second round, and the tie-break of the fusion weight (higher JPEG figure, then the smaller weight) was not in the protocol; no tie occurred.

### Second round

The *development images* of the second round are the check set and the development test set together (541 images); every decision of that round was taken on them. The two versions of the project's CNN are called setting 4 (first round), with the default threshold, and setting 5 (published), with the threshold tuned on the development images.

1. **Ranking on the final test set (384 images).** Claude Sonnet 5.5 96.6%, the combination 95.3%, the Community Forensics detector 92.2%, setting 5 (published) 81.5%, the four other published detectors 50.0% to 75.8%. Primary comparisons: the combination is more accurate than the Community Forensics detector as distributed (p = 0.004); setting 5 (published) cannot be told apart from setting 4 (first round) (82.3%, p = 0.791).
2. **Secondary comparison: the published detector with a tuned threshold.** With its threshold chosen on the development images by the same rule as the project's own detectors, the Community Forensics detector reaches 94.0%. The combination is not significantly better than that (p = 0.302). Much of the advantage of the combination over the detector as distributed therefore comes from the threshold; what the CNN adds beyond it is not significant.
3. **JPEG re-compression at quality 90.** The Community Forensics detector falls from 92.2% to 82.3% (tuned threshold: 88.5%), the combination from 95.3% to 90.1%, setting 5 (published) from 81.5% to 78.1%. After re-compression the combination is clearly better than the detector as distributed (p < 0.001), but not than the tuned one (p = 0.238).
4. **ChatGPT and Gemini set (168 images).** Claude Sonnet 5.5 is the most accurate (85.1%), followed by the Ateeqq detector (82.1%). The combination scores 69.0%: it accepts 71 photographs and misses 1, but recognises only 45 generated images and misses 51. The Community Forensics detector scores 57.7% and recognises 26 generated images.
5. **Reddit set (120 images).** Claude Sonnet 5.5 is again the most accurate (96.7%), followed by the Ateeqq detector (83.3%) and setting 4 (first round) (79.2%). The combination scores 70.8%: it accepts 59 photographs and misses 1, but recognises only 26 generated images and misses 34. No detector was the best on all three held-out sets.
6. **Training setting and threshold, separated.** Accuracy on the final test set, the ChatGPT and Gemini set and the Reddit set: setting 4 (first round) 82.3%, 73.2% and 79.2%; setting 4 with a tuned threshold 83.1%, 69.6% and 71.7%; setting 5 with the default threshold 82.8%, 64.9% and 68.3%; setting 5 (published) 81.5%, 68.5% and 68.3%. On the final test set the four are close (setting 5 against setting 4, both at the default threshold: p = 0.888). On the two other sets setting 4 (first round) is the most accurate of the four, and setting 5 at the default threshold is less accurate than it (p = 0.020 and p = 0.002). On these two sets the change of training setting lowered accuracy; the tuned threshold lowered that of setting 4 as well, and changed that of setting 5 little.
7. **An answer "generated" was more reliable than an answer "real".** When the combination called an image generated, it was right for 98% of those images on the final test set, 98% on the ChatGPT and Gemini set and 96% on the Reddit set. When it called an image real, it was right for 93%, 58% and 63%. On new kinds of image, "real" means only that no known trace was found.
8. **Variation between training runs.** Setting 5 was trained three times, differing only in the seed, each run with its own threshold chosen by the same rule (0.69, 0.38 and 0.30 on the "AI" score). On the final test set the runs score 81.5% (seed 42, published), 83.6% (seed 1) and 83.9% (seed 2): mean 83.0%, standard deviation 1.3 points. The published run was fixed in advance and is the least accurate of the three there.
9. **Limits.** The sets are small: the 95% confidence interval of the combination's accuracy is 92.7 to 97.0% on the final test set, 61.7 to 75.5% on the ChatGPT and Gemini set and 62.2 to 78.2% on the Reddit set. In the Reddit set the class of an image is the kind of subreddit it was posted in, so some labels may be wrong. Claude Sonnet 5.5 was asked once per image and saw JPEG copies; whether any test image was in its training data, or in that of the published detectors, is not known. Three training runs give only a rough estimate of the variation between runs (standard deviation 1.3 points on the final test set), so differences of one or two points between single runs of the CNN are not meaningful.
