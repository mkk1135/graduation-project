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
