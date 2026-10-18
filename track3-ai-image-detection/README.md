# 🕵️ Track 3 — AI-Generated Image Detection 🏆

<p>
  <img alt="Prize: $300" src="https://img.shields.io/badge/%F0%9F%8F%86%20Prize-%24300-FFD700?style=flat-square">
  <img alt="Device: Snapdragon 8 Elite Gen 5" src="https://img.shields.io/badge/Device-Snapdragon%208%20Elite%20Gen%205-3253DC?style=flat-square&logo=qualcomm&logoColor=white">
  <img alt="Model: Qwen2-VL-2B" src="https://img.shields.io/badge/Model-Qwen2--VL--2B-615CED?style=flat-square">
  <img alt="Runtime: Genie on HTP" src="https://img.shields.io/badge/Runtime-Genie%20%2F%20QNN%20HTP-informational?style=flat-square">
  <img alt="First qualified submission" src="https://img.shields.io/badge/%F0%9F%A5%87-first%20qualified%20submission-success?style=flat-square">
</p>

## 🎯 The task

Decide whether an image is real or AI-generated, and explain the decision.

**Explanation criteria.** The model judges the image on eight criteria:
- Lighting & shadows
- Edges & boundaries
- Texture & resolution
- Perspective
- Physical & common-sense logic
- Text & symbols
- Human & biological structure
- Material & object details

**Prompts.** Evaluation runs two fixed prompts:
1. The first prompt asks for step-by-step analysis of the image.
2. The second prompt turns that analysis into JSON. The JSON holds an `overall_likelihood` plus a 0/1 `score` and an `evidence` string for each criterion.

**Requirements:**
- **Device:** Snapdragon 8 Elite Gen 5. The model runs on the HTP through Qualcomm Genie.
- **Validity:** Generation must exceed 15 tokens/s, and each prompt's output must be at most 500 tokens.

**Scoring:**
- **Real images:** The score is the DetectionScore.
- **AI-generated images:** `0.5 × DetectionScore + 0.5 × ExplanationScore`.
- **ExplanationScore:** `0.5 × criterion exact-match accuracy + 0.5 × evidence semantic similarity`.

**Sample solution:** Qwen2-VL-2B-Instruct, prepared with Qualcomm's "Tutorial for Qwen2_VL_2b (IoT)." A submission package contains:
- The compiled vision encoder and LLM context binaries
- Embedding weights, RoPE tables, an attention mask, and `tokenizer.json`
- An `inputs.json` with the preprocessing settings and Genie runtime config

## 📤 What I submitted

I kept the organizers' compiled binaries unchanged and only edited `inputs.json`. On March 1, 2026 at 14:41, team `lpcvc_participant` made **the first qualified submission in Track 3**. It earned **the $300 early-submission prize**.

Results from the [official leaderboard](https://lpcv.ai/2026LPCVC/leaderboard/track3):

| | Score | TPS |
|---|---|---|
| Baseline | 0.397 | 21.74 |
| **Mine** (3/1/2026) | **0.397** (same) | 20.33 |

The submitted config is [`submission/inputs.json`](submission/inputs.json). Compared with the sample solution's `contestant_uploads/inputs.json`:

| Setting | Baseline | Mine |
|---|---|---|
| `engine.n-threads` | 3 | **4** |
| `QnnHtp.cpu-mask` | `0xe0` (cores 5–7) | **`0xf0` (cores 4–7)** |
| `dialog.max-num-tokens` | 1024 | **512** |
| `sampler.temp` | 0.1 | **0.15** |
| `sampler.top-p` | 0.1 | **0.15** |

Other teams resubmitted the unchanged baseline and measured anywhere from about 18 to 28 TPS. So the TPS difference between my entry and the baseline is within run-to-run noise, and these settings didn't produce a measurable speedup. The 512-token cap does match the 500-token output limit.

## 🔁 Reproduce

1. Download the full sample-solution package, about 2.4 GB of binaries, from the link in the [upstream README](https://github.com/lpcvai/26LPCVC_Track3_Sample_Solution).
2. Replace its `inputs.json` with `submission/inputs.json`.
3. Submit the package following the upstream instructions.
