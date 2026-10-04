# MCQ Branch Progress

## Setup

- MCQ train/eval datasets were derived deterministically from the dense exact-answer dataset.
- Train root: `outputs/wiki_snapshot_phase1/runs/mcq_from_dense_q8_115articles`
- Train rows: `920`
- Eval rows: `200`
- MCQ answer format: single capital letter `A`/`B`/`C`/`D`

## Important Fixes

- Fixed `exact_match` in [src/still/eval/common.py](/mnt/ssd1/shreyansh/home_dir/STILL/src/still/eval/common.py) so normalized gold answers are compared correctly. This mattered for MCQ letter evaluation.
- Found that the frozen teacher still assigns high next-token probability to `<think>` on MCQ prompts, even when full-context decoding eventually reaches the right letter.
- Added `kl_weight` to STILL training so MCQ runs can disable KL and train only on the exact target letter. This avoids distilling the teacher's reasoning-token preference into the student.

## Results

| Method | Budget | Objective | Accuracy |
|---|---:|---|---:|
| Full context | full | n/a | `0.95` |
| Truncation | `512` tokens | n/a | `0.705` |
| Truncation | `1024` tokens | n/a | `0.775` |
| STILL | `512` latents | `KL + CE` | `0.0` |
| STILL | `1024` latents | `KL + CE` | `0.0` |
| STILL | `512` latents | `CE only` | `0.205` |
| STILL | `1024` latents | `CE only` | `0.285` |
| STILL | `512` latents | `Option CE only` | `0.0` |

## Conclusions

- The MCQ task is valid locally: full-context accuracy is high.
- The original MCQ STILL objective was broken by KL against teacher logits that preferred `<think>`.
- Removing KL produces the first real quality signal:
  - `512` latents: `0.205`
  - `1024` latents: `0.285`
- A more explicit option-restricted loss on the first answer token did **not** help. It collapsed into garbage and scored `0.0`, so the simple first-token 4-way classifier is not the right training objective for this model family.
- Accuracy improves with latent budget under the CE-only objective, which is directionally aligned with the Baseten scaling story.
- STILL is still substantially worse than naive truncation at the same order-of-magnitude memory budget, so the implementation is not yet competitive.
