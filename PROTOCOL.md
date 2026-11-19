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
