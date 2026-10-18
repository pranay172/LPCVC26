# 🖼️ Track 1 — Image-to-Text Retrieval

<p>
  <img alt="Device: XR2 Gen 2" src="https://img.shields.io/badge/Device-XR2%20Gen%202-3253DC?style=flat-square&logo=qualcomm&logoColor=white">
  <img alt="Model: OpenAI CLIP" src="https://img.shields.io/badge/Model-OpenAI%20CLIP-412991?style=flat-square">
  <img alt="Metric: Recall@10" src="https://img.shields.io/badge/Metric-Recall%4010-blue?style=flat-square">
  <img alt="Latency: 20.9 ms vs 30.7 ms" src="https://img.shields.io/badge/Latency-20.9%20ms%20(baseline%2030.7)-brightgreen?style=flat-square">
  <img alt="Status: submitted" src="https://img.shields.io/badge/Status-submitted-success?style=flat-square">
</p>

## 🎯 The task

Given an image, retrieve the most relevant text descriptions from a candidate pool.

- **Model:** The submission is two compiled encoders.
  - **Image encoder:** Takes a `1×3×224×224` float32 RGB image, already resized and divided by 255. No mean/std normalization is applied, so any normalization has to be built into the model.
  - **Text encoder:** Takes `1×77` int32 token IDs from the `openai/clip-vit-base-patch32` tokenizer.
  - Both encoders output embeddings. The evaluator ranks every text against every image by cosine similarity.
- **Data:** About 300 hidden test images, each with several descriptions. The organizers released about 50 sample images.
- **Device:** Qualcomm XR2 Gen 2 (`"XR2 Gen 2 (Proxy)"` on AI Hub).
- **Validity:** The image and text encoders must run in under 35 ms combined. At launch, the rule was "faster than the sample solution."
- **Ranking:** Recall@10.
- **Early prize:** The organizers clarified on Slack that the $300 early prize required beating the sample solution on both accuracy and latency.
- **Sample solution:** OpenAI-CLIP ViT-B/16.

## 📤 What I submitted

On March 1, 2026, the first day of the competition, I submitted the unchanged ViT-B/16 baseline, recompiled, as team `lpcvc_participant`. I made two changes, captured in [`submission/baseline_recompile.patch`](submission/baseline_recompile.patch), which applies to the pinned submodule commit:

- **Compile target:** `--target_runtime qnn_context_binary` instead of `qnn_lib_aarch64_android`.
- **ONNX export:** The TorchScript exporter at opset 17 instead of `dynamo=True` at opset 18.

Results from the [official leaderboard](https://lpcv.ai/2026LPCVC/leaderboard/track1):

| | Recall@10 | Total time | Image encoder | Text encoder |
|---|---|---|---|---|
| Baseline | 0.4923 | 30.71 ms | 26.16 ms | 4.56 ms |
| **Mine** (3/1/2026 16:23) | **0.4923** (same) | **20.91 ms** (32% faster) | 16.37 ms | 4.54 ms |

The accuracy is identical because the model is identical. The entire speedup comes from the compile target. The entry beat the baseline on latency but only tied it on accuracy, so it didn't qualify for the Track 1 early prize.

## 🔬 Follow-up: CLIP ViT-B/32

The scripts in this folder swap in **CLIP ViT-B/32**:
- At 224×224 it handles 49 image patches instead of 196, cutting the image encoder's attention cost by about 4×.
- It keeps the same 512-d embedding space and the same tokenizer, so nothing else in the pipeline changes.

| Script | Purpose |
|---|---|
| `export_onnx.py` | Loads `clip.load("ViT-B/32")` and exports separate image and text encoders to `exported_onnx/` |
| `compile_and_profile.py` | Validates the ONNX files, compiles both encoders for XR2 Gen 2, and profiles them |
| `upload_dataset.py` | Preprocesses the sample images and tokenizes the texts, then uploads both to AI Hub |
| `inference.py` | Runs both encoders on the device and computes Recall@10 |
| `setup_env.sh` | A conda environment that resolved cleanly: Python 3.10, torch 2.4.1+cu121, qai-hub 0.45, qai-hub-models 0.47, and transformers < 5 |

AI Hub inference jobs on XR2 Gen 2 measured 7.4 ms for the image encoder and 4.6 ms for the text encoder, each a single-iteration run. I never finished a Recall@10 comparison for this variant, so I didn't submit it.

⚠️ **Known gap:** Neither `upload_dataset.py` nor the exported image encoder applies CLIP's mean/std normalization. The official evaluator doesn't normalize either, so normalization should be built into `ImageEncoderWrapper`. That's probably an easy accuracy gain.

## 🚀 Run

1. Set up the environment:

   ```bash
   bash setup_env.sh
   ```

   ```bash
   conda activate lpcvc1
   ```

2. Put the [sample dataset](https://lpcv.ai/2026LPCVC/tracks/track1) in `dataset/images/`, `dataset/img_list.csv`, and `dataset/txt_list.csv`.

3. Export the encoders:

   ```bash
   python export_onnx.py
   ```

4. Compile and profile them. This prints the compile job IDs.

   ```bash
   python compile_and_profile.py
   ```

5. Upload the dataset. This prints the dataset IDs.

   ```bash
   python upload_dataset.py
   ```

6. Paste the compile job IDs and dataset IDs into the `tasks` dict in `inference.py`, then run it:

   ```bash
   python inference.py
   ```
