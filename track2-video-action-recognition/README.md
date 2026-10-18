# 🏋️ Track 2 — Action Recognition in Video

<p>
  <img alt="Device: Dragonwing IQ-9075" src="https://img.shields.io/badge/Device-Dragonwing%20IQ--9075-3253DC?style=flat-square&logo=qualcomm&logoColor=white">
  <img alt="Model: R(2+1)D-18" src="https://img.shields.io/badge/Model-R(2%2B1)D--18-EE4C2C?style=flat-square&logo=pytorch&logoColor=white">
  <img alt="Dataset: QEVD" src="https://img.shields.io/badge/Dataset-QEVD-orange?style=flat-square">
  <img alt="Status: not submitted" src="https://img.shields.io/badge/Status-not%20submitted-lightgrey?style=flat-square">
</p>

## 🎯 The task

Classify the exercise being performed in a short video clip from the Qualcomm Exercise Video Dataset (QEVD, from "What to Say and When to Say it: Live Fitness Coaching as a Testbed for Situated Interaction," NeurIPS 2024). The hidden test set is drawn from QEVD as well.

- **Input:** The evaluator samples a fixed number of frames uniformly from each 2–10 s video. It resizes them to 128×171, center-crops to 112×112, normalizes them with the Kinetics mean/std, and stacks them into a `(1, 3, T, 112, 112)` clip. `T` was 8 frames at launch and was later changed to 16. Any other preprocessing has to be built into the model.
- **Output:** Class logits.
- **Device:** Qualcomm Dragonwing IQ-9075 EVK.
- **Validity:** Execution time under 34 ms.
- **Ranking:** Accuracy.
- **Sample solution:** R(2+1)D-18 from `torchvision`, fine-tuned on QEVD by the organizers. It outputs 92 classes from 8-frame clips. The checkpoint `model_29.pth` is linked from the [sample solution README](https://github.com/lpcvai/26LPCVC_Track2_Sample_Solution).

## 🛠️ What I did

I worked on getting a correct, measurable baseline running on the device before trying new models. I never submitted to this track.

1. **Data prep.** Both notebooks in `notebooks/` run on Colab with Google Drive mounted.
   - `qevd_data_download.ipynb` downloads the QEVD-FIT-300k parts from the Qualcomm Software Center. The zips need a `zip -FF` repair before they'll extract.
   - `refactor_dataset.ipynb` runs the sample solution's `DatasetRefactorer`, which sorts the raw clips into class folders using `fine_grained_labels_release.json`.

2. **`recompile_baseline.py`** exports the fine-tuned checkpoint correctly.
   - My first attempt loaded the checkpoint into `qai_hub_models.models.resnet_2plus1d.Model` and swapped in a 92-class `fc` head. The resulting binary was still the original Kinetics-400 network, because the wrapper's `forward()` doesn't go through the patched `model.model`. Watch for this trap if you use `qai_hub_models` wrappers with your own weights.
   - The fix is a standalone `NormalizedR2Plus1D` module. It loads the 92-class backbone with `strict=True` and builds the Kinetics normalization into the graph, so it expects raw `[0, 1]` frames.
   - The script exports the module to ONNX at opset 17, compiles a `qnn_context_binary` for the IQ-9075, downloads the `.bin`, and shares the job with the organizers.

3. **`evaluate_baseline.py`** runs end-to-end evaluation on the device.
   - Decodes each validation `.mp4` and samples 8 frames uniformly.
   - Resizes the frames to 128×171, center-crops them to 112×112, and converts them to channel-last (NTHWC), the layout the compiled binary expects.
   - Uploads the tensors to AI Hub in chunks of 538 to stay under the 2 GB dataset limit.
   - Runs inference jobs and reports Top-1 and Top-5 accuracy.

⚠️ **Caveat:** The official evaluator feeds clips that are already normalized. `recompile_baseline.py` normalizes again inside the model, and it uses 8 frames instead of the 16 frames in the final rules. Before submitting, I'd have to remove the built-in normalization and retrain or export for 16 frames.

## 🚀 Run

You need `torch`, `torchvision` (with `av` for video decoding), `onnx`, `h5py`, and `qai-hub`.

1. Put `model_29.pth` at `model_checkpoints/model_29.pth`.
2. Put the validation videos at `dataset/val/<class_name>/*.mp4`.
3. Compile the model:

   ```bash
   python recompile_baseline.py
   ```

4. Run the evaluation:

   ```bash
   python evaluate_baseline.py
   ```

`evaluate_baseline.py` reads `class_map.json` from the `26LPCVC_Track2_Sample_Solution/` submodule.
