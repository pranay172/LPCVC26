# ⚡ LPCVC 2026 — My Work on All Three Tracks

<p align="center">
  <a href="track3-ai-image-detection/"><img alt="Prize: $300 (Track 3)" src="https://img.shields.io/badge/%F0%9F%8F%86%20Prize-%24300%20%C2%B7%20Track%203-FFD700?style=for-the-badge"></a>
  <a href="https://lpcv.ai/"><img alt="IEEE LPCVC 2026" src="https://img.shields.io/badge/IEEE-LPCVC%202026-00629B?style=for-the-badge&logo=ieee&logoColor=white"></a>
  <a href="https://aihub.qualcomm.com/"><img alt="Qualcomm AI Hub" src="https://img.shields.io/badge/Qualcomm-AI%20Hub-3253DC?style=for-the-badge&logo=qualcomm&logoColor=white"></a>
</p>
<p align="center">
  <img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="PyTorch 2.4" src="https://img.shields.io/badge/PyTorch-2.4-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="ONNX" src="https://img.shields.io/badge/ONNX-opset%2017-005CED?style=flat-square&logo=onnx&logoColor=white">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"></a>
</p>

My code and notes from the [2026 IEEE Low-Power Computer Vision Challenge (LPCVC)](https://lpcv.ai/), organized by Purdue and sponsored by Qualcomm. Each track asks for a model that is both accurate and fast on a specific Qualcomm edge device. Models are compiled, profiled, and evaluated through [Qualcomm AI Hub](https://aihub.qualcomm.com/).

The competition window ran from March 1 to April 30, 2026. Each track paid $6,000, $3,000, and $1,000 for the top three places, plus **$300 each to the first five teams with a valid submission that beat the sample solution**.

## 🎯 The three tracks

🖼️ **[Track 1: Image-to-Text Retrieval](track1-image-text-retrieval/)**
- **Task:** Given an image, retrieve its matching text descriptions from a pool of candidates. The model outputs separate image and text embeddings, and the evaluator ranks the texts by cosine similarity. The hidden test set has about 300 images.
- **Device:** XR2 Gen 2.
- **Valid if:** The image and text encoders together run in under 35 ms.
- **Ranked by:** Recall@10.
- **Sample solution:** OpenAI-CLIP.

🏋️ **[Track 2: Action Recognition in Video](track2-video-action-recognition/)**
- **Task:** Classify the exercise being performed in a short clip from Qualcomm's QEVD fitness-video dataset.
- **Device:** Dragonwing IQ-9075 EVK.
- **Valid if:** Inference takes under 34 ms.
- **Ranked by:** Top-1 accuracy.
- **Sample solution:** R(2+1)D-18.

🕵️ **[Track 3: AI-Generated Image Detection](track3-ai-image-detection/)**
- **Task:** Decide whether an image is real or AI-generated, and explain why across eight criteria, such as lighting, text, and anatomy. The explanation is structured JSON from a two-prompt vision-language pipeline.
- **Device:** Snapdragon 8 Elite Gen 5.
- **Valid if:** Generation runs faster than 15 tokens/s.
- **Ranked by:** A combination of detection accuracy and explanation quality.
- **Sample solution:** Qwen2-VL-2B.

## 🛠️ What I did

| Track | Work | Outcome |
|---|---|---|
| 🖼️ 1 | Recompiled the CLIP baseline for a faster runtime. Built a CLIP ViT-B/32 variant with 4× fewer image patches. | Day-1 submission: same Recall@10 (0.4923), 20.9 ms vs. 30.7 ms for the baseline. The ViT-B/32 variant was never submitted. |
| 🏋️ 2 | Correctly exported the fine-tuned R(2+1)D checkpoint and built an evaluation pipeline that runs on the device | No submission |
| 🕵️ 3 | Tuned the Genie runtime and sampler settings of the Qwen2-VL baseline | 🏆 **$300 prize.** First qualified Track 3 submission: score 0.397, same as the baseline, at 20.33 TPS vs. 21.74 for the baseline. |

Each track's README has details and results. All scores and timings come from the [official leaderboards](https://lpcv.ai/2026LPCVC/leaderboard/track1).

## 📁 Layout

```
.
├── track1-image-text-retrieval/
│   ├── 26LPCVC_Track1_Sample_Solution/   # submodule: official sample solution (pinned)
│   ├── submission/baseline_recompile.patch  # what I actually submitted
│   ├── export_onnx.py                    # CLIP ViT-B/32 → ONNX
│   ├── compile_and_profile.py            # AI Hub compile + profile
│   ├── upload_dataset.py, inference.py   # on-device inference + Recall@10
│   ├── requirements.txt
│   └── setup_env.sh                      # conda env that worked for me
├── track2-video-action-recognition/
│   ├── 26LPCVC_Track2_Sample_Solution/   # submodule: official sample solution (pinned)
│   ├── recompile_baseline.py             # export + compile the fine-tuned checkpoint
│   ├── evaluate_baseline.py              # preprocess → on-device inference → Top-1/Top-5
│   └── notebooks/                        # Colab: download + sort the QEVD dataset
└── track3-ai-image-detection/
    ├── 26LPCVC_Track3_Sample_Solution/   # submodule: official sample solution (pinned)
    └── submission/inputs.json            # the prize-winning config
```

## 🚀 Setup

```bash
git clone --recurse-submodules https://github.com/pranay172/LPCVC26.git
```

The official sample solutions are pinned as git submodules at the commits I worked against. Datasets, checkpoints, and compiled binaries aren't included. Each track's README says where to get them.

You also need a Qualcomm AI Hub account. Configure your API token once:

```bash
qai-hub configure --api_token <YOUR_TOKEN>
```

## 🙏 Acknowledgements

The sample solutions, datasets, and evaluation protocol come from the LPCVC 2026 organizers ([lpcvai](https://github.com/lpcvai)) and Qualcomm. This repo contains only my changes and scripts on top of their work.

## 📄 License

My code and documentation are released under the [MIT License](LICENSE). The submodules point to the organizers' sample solutions, which remain their authors' work and are not covered by this license. Datasets and checkpoints are subject to their own terms.
