# Evaluation protocol for the second round

Written on 19 November 2026 and committed **before** any of the experiments below was run, so that the final numbers cannot be shaped by looking at them.

## Why a second round

On 18 November every approach was scored on the test set of 384 images (`results/main/`). Those results are known now, and they have shown where the fine-tuned CNN is weak: small, lightly compressed photographs. Any change made from here on is informed by that test set, so it can no longer serve as an unbiased final test. It becomes a **development test set**, and a new final test set is drawn.

## Sets and their roles

| Set | Files | Images | Role from now on |
|---|---|---|---|
| Training and validation | see `train.py` | 36,562 | training; the 10% validation split only monitors training |
| Check set | `results/check/` | 157 | development: choosing training settings, processing test |
| Development test set (the former main test set) | `results/main/` | 384 | development: choosing settings, threshold and fusion |
| **Final test set** | `results/final/` | 384 | held out; drawn on 19 November like the former main test set, from other images, photographers and categories |
| **ChatGPT and Gemini set** | `results/chat/` | 168 | held out; images from GPT Image 2, GPT Image 1.5, Nano Banana and Nano Banana Pro, and photographs from DOCCI and ImageNet |

No model is scored on the two held-out sets until every decision below has been taken and committed. Each approach is then scored on them once.

## Decisions, all taken on development data only

1. **Training setting 5** is setting 4 with stronger shrinking in the augmentation (seven times in ten, by any factor down to the crop size). It replaces setting 4 if all three hold:
   - its accuracy on the development test set, as downloaded, is higher than that of setting 4 (79.95%);
   - its mean accuracy over the six rows of the processing test on the check set is not lower than that of setting 4 (84.29%);
   - its accuracy on the check set, as downloaded, is not more than two points below that of setting 4 (89.17%).
2. **Decision threshold.** For the adopted CNN, the threshold on its "real" score is the one that gives the highest accuracy on the check set and the development test set taken together (541 images); among equally good thresholds the one closest to 0.5 is taken. It is built into the published model, so that the model's own answer uses it.
3. **Combination with the published Community Forensics detector.** The combined score is `w * CNN + (1 - w) * Community Forensics` on the "AI" probabilities, with `w` chosen from 0.25, 0.5 and 0.75 and the threshold chosen as in point 2. The combination is adopted as the project's system if its accuracy on the 541 development images is higher than that of each of its two parts, both as downloaded and after JPEG re-compression at quality 90. Otherwise the adopted CNN alone is the system.
4. **Repeated runs.** The adopted setting is trained three times, with seeds 42, 1 and 2. The published model is the seed-42 run, fixed here in advance, not the best of the three. The other two runs measure the variation between runs.

## Final scoring

After these decisions are committed, the following are scored once on the final test set and on the ChatGPT and Gemini set, as downloaded and after JPEG re-compression at quality 90: the five published detectors, the first CNN, setting 4 (the model published until now), the adopted CNN (three seeds), the combination if adopted, and Claude Sonnet 5.5 in a blind test (as downloaded only).

Primary comparisons, fixed in advance, with exact McNemar tests on the final test set:

- the project's system against the Community Forensics detector, the best published detector on the development test set;
- the adopted CNN (seed 42) against setting 4.

Nothing is changed after the final scoring. Whatever the outcome, it is reported.

## Addendum of 19 November 2026, before any scoring on held-out data

A third held-out set was added: the **Reddit set** (`results/wild/`, 120 images), 60 images posted in subreddits for AI-generated images and 60 posted in photography subreddits, from the `reddit` test split of OpenFake, without video frames. It tests images as they circulate online, after the platform's own compression. Its labels are the kind of subreddit an image was posted in, so some may be wrong; it is a secondary measurement and is scored once, together with the other two held-out sets. No decision depends on it.

A stronger baseline was added as a **secondary comparison**: the Community Forensics detector with its decision threshold chosen on the 541 development images by the same rule as in point 2 (`python decide.py threshold --model commfor`), instead of its published threshold of 50%. The project's own detectors have thresholds tuned on those images, so the published detector is given the same advantage. It is computed from the published detector's scores (`rethreshold.py`) and reported next to the primary comparisons.

**Separating the effect of the threshold from that of the training setting.** Point 1 compares setting 5 with setting 4 at the default threshold, and the adopted CNN then receives a tuned threshold (point 2). On the development data setting 5 recognises more photographs and fewer generated images than setting 4, a change that a different threshold also produces. To keep the two effects apart, two more variants are scored on the held-out sets as secondary measurements: setting 4 with its threshold chosen on the 541 development images by the rule of point 2 (`python decide.py threshold --model cnn_final`, computed from its stored scores with `rethreshold.py`), and setting 5 with the default threshold of 50% (the trained network before the threshold is built in). The primary comparisons are unchanged.

**Very large images.** The CNN of this project analyses an image at its original resolution, and an image of more than 8 megapixels does not fit into the memory of the 8 GB GPU used. Of such an image only the central 2,048 x 2,048 pixels are analysed, as the local interface does (`fit` in `evaluate.py`). No image of the check set, the development test set or the final test set is that large (the largest has 7.4 megapixels), so no earlier result changes and the primary comparisons are not affected. The rule applies to 1 photograph of the ChatGPT and Gemini set and to 41 images of the Reddit set (35 of its 60 photographs and 6 of its 60 images from AI subreddits). The published detectors shrink every image themselves and receive the whole image; in the combination only the CNN part is affected. This rule was added after the sizes of the held-out images had been read from the files, and before any of them was scored.

## Decisions taken on 19 November 2026

Taken by `decide.py` from the results database and recorded in `results/decisions.json`, before any image of the held-out sets was scored.

1. **Training setting.** Setting 5 against setting 4, default threshold: development test set 80.99% against 79.95% (met); mean of the processing test 85.99% against 84.29% (met); check set 87.26% against 89.17% (met). Setting 5 is adopted.
   For the record: setting 5 recognises more photographs and fewer generated images than setting 4, and once each setting has the threshold of point 2, setting 4 is the more accurate on the 541 development images (85.95% against 84.29%; after JPEG re-compression 84.47% against 81.15%). The rule of point 1 compares the settings at the default threshold and is applied as written. Setting 4 with a tuned threshold is scored on the held-out sets as a secondary measurement (see the addendum above), so that both paths can be compared on data that neither has seen.
2. **Decision threshold of the adopted CNN (seed 42).** 0.6894 on the "AI" score: 82.81% of the 541 development images correct at the default threshold, 84.29% at the chosen one (81.15% after JPEG re-compression).
3. **Combination.** Weight of the CNN 0.25, threshold 0.2510: 93.53% of the development images correct, 88.72% after JPEG re-compression. The parts alone: cnn_s5 84.29% and 81.15%, commfor 88.54% and 79.30%. The combination is adopted as the system of the project. These figures are computed from the stored scores of the two parts, which are rounded to four decimals; run on the images themselves, the combination classifies 505 of the 541 development images correctly (93.35%) and 481 after JPEG re-compression (88.91%), one image fewer and one more.
4. **Thresholds for the secondary comparisons.** Community Forensics detector: 0.1227 (88.54% of the development images correct at its published threshold, 92.05% at the chosen one). Setting 4: 0.8272 (82.62% and 85.95%).

The thresholds of the two repeated runs (seeds 1 and 2) are chosen by the same rule when those runs have been trained; they do not affect the published model, the combination or the primary comparisons.

## Final scoring (19 November 2026)

After the decisions above were committed (commit 7b8e05d), every approach was scored once on the final test set, the ChatGPT and Gemini set and the Reddit set, as downloaded and after JPEG re-compression at quality 90; Claude Sonnet 5.5 was scored in a blind test, as downloaded only. Nothing was changed afterwards. The two repeated runs (seeds 1 and 2) are trained with the committed code after this scoring, and their thresholds are chosen by the rule of point 2. The results, including the unfavourable ones, are in [COMPARISON.md](COMPARISON.md).

# Third round

Written on 21 November 2026 and committed **before** any image of this round was downloaded, any new set was drawn, any model was trained or anything was scored. Everything above is unchanged.

## Why a third round

The held-out results of the second round showed where the system is weak: images from the newest generators (ChatGPT and Gemini set: combination 69.0%, CNN 68.5%) and images as shared online (Reddit set: 70.8% and 68.3%). On both sets the system accepted almost every photograph but recognised only about half of the generated images. The third round trains on new data aimed at these images. Because all five scored sets have now been seen, they all become development data, and new held-out sets are drawn from images never used before.

## Sets and their roles

| Set | Images | Role in the third round |
|---|---|---|
| Check set, development test set, final test set (`results/check/`, `results/main/`, `results/final/`) | 157 + 384 + 384 = 925 | development, **old kinds**: generators of the Community Forensics evaluation set; photographs from Wikimedia Commons, Unsplash and Open Images |
| ChatGPT and Gemini set, Reddit set (`results/chat/`, `results/wild/`) | 168 + 120 = 288 | development, **new kinds** |
| **Final test set 3** (`results/final3/`) | 384 | held out, drawn like the final test set. 12 images of each of the 11 commercial and 6 of each of the 10 open-source generators of the Community Forensics evaluation set, the first in shard order of the 19 shards 5, 17, 31, 46, 62, 86, 158, 173, 183, 191, 208, 217, 238, 253, 263, 284, 317, 360 and 389, which no earlier set has read (1.47 GB). 64 Wikimedia Commons photographs from categories not used before (Quality images of lakes, palaces, gardens, horses, statues, boats, waterfalls and towers, 8 each; plants fills up any shortfall), one per uploader, excluding every earlier file and every author of the earlier Commons draws. 64 Unsplash Lite and 64 Open Images V7 validation photographs, seeded random samples with seed 2028, excluding every earlier URL |
| **ChatGPT and Gemini set 3** (`results/chat3/`) | 384 | held out. From the `core` test split of OpenFake v2.1, without video frames, in file order: 48 GPT Image 2 and 48 GPT Image 1.5 images, 96 DOCCI and 96 ImageNet photographs. 48 Nano Banana images, the first of `bitmind/nano-banana` file `data/train-00001-of-00031.parquet`, which no earlier set has read. 48 Nano Banana Pro images, the first generated images of the test split of `julienlucas/nanobanana-pro-dataset` after the 24 used before |
| Seedream 5.0 control (`results/control3/`) | 48 | held out, secondary. The first 48 `seedream-v5.0` images of the same OpenFake split, a generator that no training data of any round contains. Only the share recognised as generated is reported |
| **Reddit set 3** (`results/wild3/`) | 384 | held out. From the `reddit` test split of OpenFake v2.1, without video frames, in file order: 192 images from AI subreddits and 192 from photography subreddits. Labels follow the kind of subreddit, as before |

`make_test_set.py` draws the new sets. Every OpenFake row of an earlier set is skipped, and so is every row whose SHA-256 (the dataset's `image_sha256`) equals that of a row taken before; this matters because the `reddit` test split holds 1,856 rows that repeat another row's image. Within a set every image is taken once. The lists (`results/<set>/<set>_set.csv`) name the source, file, row and SHA-256 of every image. Because drive C: of the laptop is nearly full, the images of these sets, all round-3 downloads, the training data, the Hugging Face cache of the round (`HF_HOME`) and the checkpoints are kept on drive D: under `D:/ai-detector-round3/`; nothing is deleted on C:. The new sets are drawn before the training data are assembled. No model is scored on them before the decisions below are committed.

## New training data: no data of the earlier rounds

Only datasets and splits that no earlier training used, all from OpenFake v2.1 (`ComplexDataLab/OpenFake`, tag v2.1, commit 00b2f0ae, CC BY-NC 4.0). `make_train_set.py` selects them by the fixed rules below and lists every image in `results/round3/train_set.csv` (part, file, row group, row, row id, SHA-256, class, generator or subreddit). The expected numbers were computed from the datasets' metadata, read before this protocol was written; no image was downloaded.

| Part | Rule | Expected images |
|---|---|---|
| Photographs of the `core` train split | all photographs of 110 row groups chosen at random (seed 2026) among the 33,587 row groups of the split | 1,994 LAION (web images, many of people and news events), 1,848 Pexels (stock photographs) |
| Other generators of the `core` train split | 1,000 generated images drawn at random (seed 2026) from the same 110 row groups | 1,000 from about 50 generators |
| API generators of the `core` train split | single images drawn at random (seed 2026) from the rest of the split, read one by one by HTTP range requests and checked against the split's SHA-256 | 2,000: GPT Image 1 700, Imagen 4 400, Nano Banana 2 400, Grok Imagine 200, FLUX 1.1 pro 100, Ideogram 3 100, Qwen-Image 100 |
| `supplement_civitai` (posted on Civitai, 7 May to 3 November 2026) | the images of ChatGPT Images, Google's Nano Banana, Grok Imagine, OpenAI's GPT-image-1, Google Imagen 4, Qwen Image (API), Wan 2.7 Image and Ideogram 4 in the 48 row groups where they are densest | 2,013, of which 834 ChatGPT Images, 577 Grok Imagine, 448 Nano Banana |
| `supplement_reddit`, AI subreddits (posted 29 July to 3 November 2026) | the images from aiart, aiartwork, genaigallery, ai_art, midjourney, pro_ai_art, aigeneratedart and aiimages in the 48 row groups where they are densest | 2,544 |
| `supplement_reddit`, photography subreddits | 240 images of each of itookapicture, amateurphotography, wildlifephotography, photocritique, cityporn, earthporn, analog, streetphotography, foodporn, casual_photography and portraitphotography, from the row groups where that subreddit is densest | 2,640 |

Expected in all: about 13,950 images, 6,480 photographs and 7,560 generated images, about 16 GB; the download is about 27 GB.

Left out: frames taken from videos; the subreddits pics (not a photography subreddit), nanobananasfw, veo3, klingai_videos and deepdream; the OpenFake generators that are also generators of the Community Forensics evaluation set (dalle-3, midjourney-6, imagen-3.0-002, flux.1-dev, flux.1-schnell), so that the generators of final test set 3 stay unseen; every Seedream image (seedream-v4.0, seedream-v4.5, Civitai's Seedream), so that the Seedream 5.0 control stays unseen; images over 20 MB; and every image whose SHA-256 equals that of an evaluation image of any round, old or new, or whose 256-bit difference hash (of the whole image or of its central square) differs from that of an evaluation image in at most 24 bits. After both are built, a check compares every training image with every evaluation image in this way and must find no match. The photographs come from three sources of different kinds (web images, stock photographs, photographs posted on Reddit), so that no single source stands for "real". 10% of the images are held back to monitor training only.

If the download has not finished by 22 November 2026, 18:00 Dubai time, each of the six parts is cut to the first half of its list (row groups or single images, in the listed order); this is recorded.

## Candidates

All three are trained once, seed 42, batch size 32, crops of 256 px at original resolution, on the new data only (`train.py --data round3`):

- **A, continued fine-tuning:** starts from the weights of setting 5 (`checkpoints/s5_seed42/checkpoint-3087`, the published model before its threshold was built in), 2 epochs, learning rate 1e-5, augmentation of setting 5.
- **B, continued fine-tuning with social-media processing:** as A, and before the augmentation of setting 5, with probability 0.5, the processing of a sharing platform: the longer side reduced to 1080, 1280, 1440 or 2048 px if it is larger (filter chosen at random), then saved as JPEG at quality 60 to 92 with 4:2:0 chroma subsampling (four times in five) or as WebP at quality 60 to 90; with probability 0.3 a second share (reduced and saved again in the same way, quality 50 to 85); with probability 0.15 a screenshot (rescaled by a factor 0.75 to 1.5 with a bilinear filter, kept without compression).
- **C, from the base network:** starts from `facebook/convnext-tiny-224`, 3 epochs, learning rate 5e-5, augmentation of setting 5. It shows what the weights of the earlier rounds contribute.

## Decisions, all taken on the 1,213 development images

1. **Threshold of each model.** For the current CNN (setting 5 before its threshold, `cnn_s5_raw`) and for each candidate, the threshold on the "AI" score is chosen on all 1,213 development images, as downloaded, by the rule of point 2 of the second round. Every figure below is taken at that model's own threshold.
2. **Adopting a candidate.** Figures: *old* = accuracy on the 925 images of old kinds; *old AI* = share of their 472 generated images recognised; *new* = accuracy on the 288 images of new kinds; *JPEG* = accuracy on all 1,213 images after JPEG re-compression at quality 90. For the current CNN, computed on 21 November 2026 from the stored predictions: threshold 0.6594, old 83.35%, old AI 77.97%, new 68.40%, JPEG 76.67%. A candidate qualifies if all four hold:
   - new at least 5 points higher than that of the current CNN, that is at least 73.40%: it has learnt the new kinds;
   - old at most 2 points lower, at least 81.35%, and old AI at most 5 points lower, at least 72.97%: it has not forgotten the older generators and photographs;
   - JPEG at most 2 points lower, at least 74.67%.

   Among the qualifying candidates the one with the highest mean of old and new is adopted; ties go to the higher JPEG figure, then to A before B before C. If no candidate qualifies, setting 5 stays the CNN of the project and the system of the second round is kept as it is; the held-out sets are scored all the same. For the record, not part of the rule: the change of each candidate on every source and generator of the development sets, and the processing test on the check set.
3. **Threshold built in.** The threshold of point 1 of the adopted candidate is built into it (`apply_threshold.py`).
4. **Combination with the Community Forensics detector.** If a candidate is adopted: rule 3 of the second round on the 1,213 development images, with one change. The weight is 0.25, 0.5 or 0.75 and the threshold is chosen by rule 2, as before, but the weight with the highest mean of old and new is taken (not the highest accuracy on all images, which the 925 images of old kinds would dominate), and the combination is adopted if both its mean of old and new and its JPEG figure are higher than those of each of its two parts, each at its own threshold of point 1. Otherwise the adopted CNN alone is the system. For a secondary comparison the threshold of the Community Forensics detector is chosen on the same 1,213 images by rule 2.
5. **Second opinion.** The page keeps giving a firm answer only when the system and the second opinion agree; nothing in it is tuned. As a secondary measurement, for each held-out set the share of images on which the system and the Ateeqq detector (the second opinion of the public page) agree, and the accuracy on those images, are computed from the stored predictions.
6. **Repeated runs.** After the final scoring the adopted candidate is trained twice more (seeds 1 and 2) with the committed code, to measure the variation between runs; the published model is the seed-42 run, fixed here in advance.

The decisions are recorded by `decide.py` in `results/decisions.json` under the key `round3`, next to the unchanged decisions of the second round.

## Final scoring

After the decisions are committed, each of the following is scored once on final test set 3, ChatGPT and Gemini set 3, Reddit set 3 and the Seedream 5.0 control, as downloaded and after JPEG re-compression at quality 90: the system of the third round (if it differs from that of the second), the system of the second round (combination), setting 5 (published), the three candidates, each with its threshold of point 1, setting 4 (first round), the Community Forensics detector (published threshold, and the threshold of point 4), and the four other published detectors. Claude Sonnet 5.5 is not scored again.

Primary comparisons, fixed in advance, exact McNemar tests:

- the system of the third round against that of the second round on ChatGPT and Gemini set 3 and Reddit set 3 together (768 images): the aim of the round;
- the same two systems on final test set 3 (384 images): whether the older kinds of image were forgotten.

If no candidate is adopted, the two systems are the same and nothing is tested; the figures of the current system on the new sets are reported. Nothing is changed after the final scoring. Whatever the outcome, it is reported.

## Limits known in advance

- **Same platforms in training and test.** The generators of ChatGPT and Gemini set 3 are in the new training data (GPT Image 1.5 and 2 among Civitai's "ChatGPT Images", the Nano Banana family among Civitai's "Google's Nano Banana" and as OpenFake's Nano Banana 2), and its GPT images, like the Civitai training images, were posted on Civitai. Reddit set 3 comes from the same subreddits as the Reddit training images, at an earlier time. These two sets therefore measure new images of the kinds trained on, not unseen generators or platforms, and may flatter the round. Final test set 3 and the Seedream 5.0 control measure generators absent from all training data; the Nano Banana and Nano Banana Pro images of ChatGPT and Gemini set 3 come from two datasets outside OpenFake, and its photographs (DOCCI, ImageNet) from sources not in the training data.
- **Label noise.** The class of a Reddit image, in training and in Reddit set 3, is the kind of subreddit it was posted in, and the generator of a Civitai image is the one its uploader named. Some labels are wrong; OpenFake's authors expect noise at the level of a few percent.
- **Shortcuts.** In the Reddit data the generated images are much smaller than the photographs (median 1.6 against 7.7 megapixels in a sample), many generated training images are PNG while the photographs are JPEG, and the Civitai part has generated images only. The random shrinking and JPEG re-compression of the augmentation, three kinds of photograph source, final test set 3 and the per-source figures on the development data are meant to expose a model that learns such shortcuts instead of traces of generation, but cannot rule it out.
- **Forgetting.** Candidates A and B never see the old training data again; point 2 rejects a candidate that loses more than the stated margins on the old kinds.
- **Size.** With 384 images per set, the 95% confidence interval of an accuracy is about ±4 to 5 points; differences between single training runs of one or two points are not meaningful (second round: standard deviation 1.3 points between seeds).
- **Development data used repeatedly.** The five development sets have been scored many times; the choice among three candidates and the thresholds are tuned on them, so the development figures are optimistic. Only the new held-out sets measure the result.

## Change of 21 November 2026, before any download or training

The training set was reduced to about 5,000 images at the student's request; parts and rules unchanged, each part scaled proportionally, so that photographs and generated images stay about equal in number and every listed generator and subreddit is still represented. New counts (expected 4,941; download about 10 GB):

| Part | New rule and count |
|---|---|
| Photographs of the `core` train split | all photographs of 42 row groups chosen at random (seed 2026): 780 LAION, 671 Pexels |
| Other generators of the `core` train split | 350 generated images drawn at random (seed 2026) from the same 42 row groups (47 generators) |
| API generators of the `core` train split | 700: GPT Image 1 245, Imagen 4 140, Nano Banana 2 140, Grok Imagine 70, FLUX 1.1 pro 35, Ideogram 3 35, Qwen-Image 35 |
| `supplement_civitai` | 700, a fixed number per generator, each from the row groups where that generator is densest: ChatGPT Images 290, Google's Nano Banana 160, Grok Imagine 140, OpenAI's GPT-image-1 30, Google Imagen 4 20, Qwen Image (API) 20, Wan 2.7 Image 20, Ideogram 4 20 (27 row groups) |
| `supplement_reddit`, AI subreddits | the first 750 images of the listed AI subreddits in the 16 row groups where they are densest |
| `supplement_reddit`, photography subreddits | 90 of each of the 11 listed photography subreddits, from the row groups where that subreddit is densest: 990 |

In all: 2,441 photographs and 2,500 generated images. Everything else (exclusions, candidates, decisions, held-out sets, scoring, comparisons) is unchanged.

## Record of 21 November 2026: sets drawn and candidates trained, before any scoring

- **New held-out sets**, drawn by the rules above (`results/<set>/<set>_set.csv`): final test set 3, 384 images; ChatGPT and Gemini set 3, 384; Seedream 5.0 control, 48; Reddit set 3, 384. In final test set 3 three Commons categories had too few new photographers (palaces 4, gardens 1, boats 7), and plants filled up with 12, as provided. No OpenFake row or URL of an earlier set was used again, and no file occurs twice among the 2,461 evaluation images of all rounds. Their difference hashes show one related pair across sets: Nano Banana Pro image 15 of ChatGPT and Gemini set 3 is a variant of image 16 of the development ChatGPT and Gemini set (the same scene, a different pose; 19 bits apart). The rules for the held-out sets exclude identical files only, so the set is unchanged; three Seedream 5.0 control images are such variants of each other.
- **Training data** (`results/round3/train_set.csv`): of the 4,941 selected images, 17 were left out as over 20 MB (16 Nano Banana 2, 1 other) and 4 as within 24 bits of an evaluation image (3 Pexels photographs near DOCCI image 25 of ChatGPT and Gemini set 3, 1 image of r/genaigallery near Reddit set 3 AI image 159); 4,920 remain, 2,438 photographs and 2,482 generated images (`results/round3/train_left_out.csv`). A second comparison of all 4,920 with all 2,461 evaluation images found no identical file and no pair within 24 bits. Contrary to the sentence of the change above, the rule for the AI subreddits ("the first 750 images in the 16 row groups where they are densest") gives images of six of the eight listed subreddits, none of r/midjourney and r/aigeneratedart; the rule was applied as written.
- **Candidates A, B and C** were trained as stated (`results/training/r3_A.json`, `r3_B.json`, `r3_C.json`), with the code of commit 18146cd. No model has been scored on any set in this round.

## Decisions of the third round (21 November 2026), before any scoring of the new held-out sets

The rules were coded in `decide.py` and committed with tests (commit fbd0010) before any candidate was scored. The three candidates, before their thresholds, were then scored on the 1,213 development images, as downloaded and after JPEG re-compression at quality 90. `python decide.py round3` and `python decide.py round3_fusion` took the decisions from the results database; they are recorded in `results/decisions.json` (keys `round3`, `round3_fusion`) and reproduced by `python decide.py check`.

1. **Thresholds** (rule 2 on all 1,213 images). Current CNN (setting 5 before its threshold) 0.6594, as in the protocol; candidate A 0.7504, B 0.8072, C 0.6902. For the secondary comparison, the Community Forensics detector: 0.0477.
2. **Adopting a candidate.** Limits: new at least 73.40, old at least 81.35, old AI at least 72.97, JPEG at least 74.67.

   | Model | Old (925) | Old AI (472) | New (288) | JPEG (1,213) | Mean of old and new | Qualifies |
   |---|---|---|---|---|---|---|
   | Current CNN (setting 5) | 83.35 | 77.97 | 68.40 | 76.67 | 75.88 | (baseline) |
   | Candidate A | 83.14 | 76.69 | 79.86 | 79.39 | 81.50 | yes |
   | Candidate B | 82.59 | 74.58 | 81.94 | 79.06 | 82.27 | yes |
   | Candidate C | 72.65 | 60.59 | 86.46 | 74.44 | 79.55 | no: old, old AI and JPEG below the limits |

   **Candidate B is adopted** (the higher mean of old and new of the two that qualify). Candidate C, trained from the base network on the new data only, is the most accurate on the new kinds but has forgotten much of the old ones: on the 120 generated images of open-source generators it recognises 36, setting 5 74. For the record, at each model's threshold, images correct on the development data (setting 5 / A / B / C): GPT Image 1.5 10 / 18 / 18 / 20 of 24, GPT Image 2 13 / 19 / 19 / 21 of 24, Nano Banana Pro 9 / 14 / 14 / 20 of 24, Reddit AI subreddits 26 / 43 / 43 / 48 of 60, Reddit photography subreddits 56 / 55 / 57 / 56 of 60, commercial generators of the old sets 294 / 293 / 288 / 250 of 352, open-source generators 74 / 69 / 64 / 36 of 120, photographs of the old sets 403 / 407 / 412 / 386 of 453. Processing test on the check set, at the default threshold (`results/check/diagnosis_round3.txt`), mean accuracy over its six rows: setting 5 86.0%, A 86.1%, B 84.5%, C 72.9%; candidate B accepts only 36 of 69 photographs after a slight blur (radius 0.6).
3. **Threshold built in.** 0.8072 is built into candidate B (`models/cnn_r3`); scored on the development data, it gives exactly the figures of candidate B above.
4. **Combination.** Weight 0.25: threshold 0.2191, old 94.49, new 78.12, mean 86.31, JPEG 86.15; weight 0.5: threshold 0.2804, old 92.43, new 83.68, mean 88.06, JPEG 87.47; weight 0.75: threshold 0.2881, old 88.65, new 82.29, mean 85.47, JPEG 84.42. **Weight 0.5 is taken**, and the combination is adopted: its mean of old and new (88.06) and its JPEG figure (87.47) are higher than those of candidate B (82.27, 79.06) and of the Community Forensics detector at its threshold of 0.0477 (81.46, 82.52). **The system of the third round is the combination of candidate B (weight 0.5) and the Community Forensics detector, with threshold 0.2804** (`evaluate.py fusion3:models/cnn_r3`).
5. **Repeated runs.** Candidate B is trained again with seeds 1 and 2 after the final scoring.

As decided earlier, Nano Banana Pro image 15 of ChatGPT and Gemini set 3 and the three related Seedream 5.0 images stay in the held-out sets: their rules exclude identical files only. In the final scoring the published detectors are read from the same local copies (Hub revisions) as in the second round, without downloading.

## Final scoring of the third round (21 November 2026)

After the decisions above were committed (commit 6a47cc0), every approach named in the protocol was scored once on final test set 3, ChatGPT and Gemini set 3, the Seedream 5.0 control and Reddit set 3, as downloaded and after JPEG re-compression at quality 90 (`results/final3/`, `results/chat3/`, `results/control3/`, `results/wild3/`). The candidates received their thresholds of decision 1, and the Community Forensics detector its threshold of decision 4, from their stored scores (`rethreshold.py --round3`). Nothing was changed afterwards. Accuracy with its 95% Wilson interval; exact McNemar tests on the same images.

**Primary comparisons**, the system of the third round (combination of candidate B and the Community Forensics detector) against that of the second round:

| Images | Third round | Second round | Right only with the third / only with the second | p |
|---|---|---|---|---|
| ChatGPT and Gemini set 3 and Reddit set 3 (768) | 82.9% (80.1 to 85.4) | 73.7% (70.5 to 76.7) | 102 / 31 | 4.9e-10 |
| The same, after JPEG 90 | 82.2% | 70.3% | 115 / 24 | 1.9e-15 |
| Final test set 3 (384) | 92.4% (89.4 to 94.7) | 94.5% (91.8 to 96.4) | 6 / 14 | 0.115 |
| The same, after JPEG 90 | 90.6% | 89.8% | 15 / 12 | 0.70 |

The third round reached its aim on the new kinds of image (ChatGPT and Gemini set 3 81.8% against 75.5%; Reddit set 3 84.1% against 71.9%). On final test set 3 the new system is 2.1 points less accurate; the difference is not significant. It recognises more generated images there (93.2% against 90.6%) but accepts fewer photographs (91.7% against 98.4%: Open Images 59 against 64, Unsplash 60 against 63, Wikimedia Commons 57 against 62, of 64 each); in ChatGPT and Gemini set 3 it accepts 76 of 96 ImageNet photographs against 96.

**Secondary measurements.**
- The CNN alone (candidate B with its threshold) against setting 5 (published): final test set 3 82.6% against 82.8% (p = 1.0); ChatGPT and Gemini set 3 77.6% against 71.9% (p = 0.006); Reddit set 3 83.1% against 73.2% (p = 7e-8). Seedream 5.0 control, generated images recognised: 38 of 48 against 19 (combinations: 41 against 21).
- Candidate C, which the rule rejected for forgetting, is the most accurate approach on ChatGPT and Gemini set 3 (89.1%) and Reddit set 3 (85.9%), and the least accurate CNN on final test set 3 (73.7%), as on the development data. Candidate A: 82.0%, 76.6% and 82.8%.
- The Community Forensics detector with the threshold of decision 4: final test set 3 91.9%, ChatGPT and Gemini set 3 73.7%, Reddit set 3 69.0%. The Ateeqq detector: 77.3%, 81.0% and 82.0%, and 44 of the 48 Seedream 5.0 images.
- Second opinion: the system of the third round and the Ateeqq detector agree on 74.5% of final test set 3, 72.7% of ChatGPT and Gemini set 3 and 74.0% of Reddit set 3; where they agree, the answer is right for 96.9%, 93.2% and 94.7% of the images (system of the second round: agreement 76.0%, 71.1%, 63.3%; right 97.3%, 89.7%, 92.6%).

All figures, for every approach, set and condition, are in the result files and in `results/results.db`. Limits as stated in advance: the generators of ChatGPT and Gemini set 3 and the subreddits of Reddit set 3 are in the new training data, so these two sets measure new images of the kinds trained on; final test set 3 and the Seedream 5.0 control measure generators absent from all training data.
