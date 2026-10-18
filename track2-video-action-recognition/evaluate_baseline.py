"""
LPCVC Track 2 — Baseline Evaluation Script
===========================================
Full pipeline in one script:
  1. Preprocess raw .mp4 clips → .npy tensors + manifest.jsonl
  2. Upload tensors to QAI Hub and run on-device inference with the compiled .bin
  3. Report Top-1 and Top-5 accuracy

Usage:
    cd track2-video-action-recognition
    python evaluate_baseline.py

Normalization note
------------------
Normalization (Kinetics mean/std) is BAKED INTO the compiled .bin model via the
NormalizedR2Plus1D wrapper in recompile_baseline.py. This script sends raw
[0, 1] float32 tensors to the model. This avoids the layout-dependent
normalization issues seen in the default qai_hub_models wrapper.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torchvision.io as tvio
import torchvision.transforms.functional as TF
import h5py
import qai_hub

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRACK2_ROOT = Path(__file__).parent
SAMPLE_SOL  = TRACK2_ROOT / "26LPCVC_Track2_Sample_Solution"

DATA_ROOT   = TRACK2_ROOT / "dataset" / "val"       # raw val videos
OUT_ROOT    = TRACK2_ROOT / "preprocessed_val"       # output .npy tensors
DLC_PATH    = TRACK2_ROOT / "export_assets" / "resnet_2plus1d_baseline_ep29_onnx.bin"
CLASS_MAP   = SAMPLE_SOL  / "class_map.json"
OUTPUT_H5   = TRACK2_ROOT / "dataset-export.h5"

CLIP_LEN        = 8      # frames per clip (matches training)
FRAME_RATE      = 4      # fps used during training
DEVICE_NAME     = "Dragonwing IQ-9075 EVK"
IS_CHANNEL_LAST = True   # QNN_CONTEXT_BINARY from recompile_baseline.py expects NTHWC input
CHUNK_SIZE      = 538    # stay under QAI Hub 2 GB flatbuffer limit

# Derived spatial dims
T, H, W = CLIP_LEN, 112, 112

# ---------------------------------------------------------------------------
# Video preprocessing (mirrors video_processing.py without local torchvision)
# ---------------------------------------------------------------------------

def process_video(video_path: str, clip_len: int = 8, frame_rate: int = 4) -> np.ndarray:
    """
    Decode video, uniformly sample `clip_len` frames, resize 128×171,
    centre-crop 112×112, return (1, 3, T, 112, 112) in [0, 1].

    Normalization is baked into the model, so we send raw [0, 1] floats.
    """
    vframes, _, info = tvio.read_video(video_path, pts_unit="sec", output_format="TCHW")
    # vframes: (N, C, H, W) uint8

    if vframes.shape[0] == 0:
        raise ValueError(f"Could not decode any frames from {video_path}")

    n_frames = vframes.shape[0]

    # Uniform sampling: pick clip_len frames evenly spaced over the full video
    indices = torch.linspace(0, n_frames - 1, steps=clip_len).long()
    clip = vframes[indices]  # (T, C, H, W) uint8

    # Spatial transforms: uint8 → [0,1] → resize → crop
    clip = clip.float() / 255.0                           # (T, C, H, W) in [0,1]
    clip = TF.resize(clip, [128, 171], antialias=False)   # (T, C, 128, 171)
    clip = TF.center_crop(clip, [112, 112])               # (T, C, 112, 112)

    # Permute to (C, T, H, W) then add batch dim → (1, C, T, H, W)
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0)
    return clip.numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Step 1 — Preprocess videos → .npy tensors + manifest.jsonl
# ---------------------------------------------------------------------------

def preprocess(data_root: Path, out_root: Path) -> Path:
    manifest_path = out_root / "manifest.jsonl"
    videos = sorted(data_root.rglob("*.mp4"))

    if manifest_path.exists():
        existing = sum(1 for _ in manifest_path.open())
        if existing == len(videos):
            print(f"[Preprocess] {existing} tensors already cached — skipping.")
            return manifest_path
        print(f"[Preprocess] Cache mismatch ({existing} vs {len(videos)}) — redoing.")

    if not videos:
        raise FileNotFoundError(f"No .mp4 files found under {data_root}")

    print(f"[Preprocess] Processing {len(videos)} videos ...")
    out_root.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w", encoding="utf-8") as mf:
        for video_path in videos:
            rel      = video_path.relative_to(data_root)
            label    = rel.parts[0]
            out_path = out_root / rel.with_suffix(".npy")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            clip = process_video(str(video_path), clip_len=CLIP_LEN, frame_rate=FRAME_RATE)
            np.save(str(out_path), clip)

            mf.write(json.dumps({
                "video_path":  str(video_path),
                "label":       label,
                "tensor_path": str(out_path),
                "shape":       list(clip.shape),
                "dtype":       str(clip.dtype),
            }) + "\n")
            print(f"  {label}/{video_path.name} → shape {clip.shape}")

    print(f"[Preprocess] Done — tensors in {out_root}")
    return manifest_path


# ---------------------------------------------------------------------------
# Step 2 — Run on-device inference via QAI Hub
# ---------------------------------------------------------------------------

def load_tensors(out_root: Path) -> list[np.ndarray]:
    paths = sorted(out_root.rglob("*.npy"))
    tensors = []
    for p in paths:
        x = np.load(str(p)).astype(np.float32)  # (1, 3, T, H, W)
        # Enforce frame count in case of mismatch
        cur_t = x.shape[2]
        if cur_t < T:
            x = np.pad(x, ((0,0),(0,0),(0,T-cur_t),(0,0),(0,0)), mode="edge")
        elif cur_t > T:
            x = x[:, :, :T, :, :]
        if IS_CHANNEL_LAST:
            x = np.transpose(x, (0, 2, 3, 4, 1))
        tensors.append(x)
    return tensors


def run_inference(dlc_path: Path, tensors: list[np.ndarray], output_h5: Path) -> None:
    if output_h5.exists():
        output_h5.unlink()

    device = qai_hub.Device(DEVICE_NAME)

    print(f"[Inference] Uploading compiled model: {dlc_path.name} ...")
    target_model = qai_hub.upload_model(str(dlc_path))
    print(f"[Inference] Model uploaded → {target_model}")

    print(f"[Inference] Submitting {len(tensors)} samples in chunks of {CHUNK_SIZE} ...")
    jobs = []
    for i in range(0, len(tensors), CHUNK_SIZE):
        chunk   = tensors[i : i + CHUNK_SIZE]
        dataset = qai_hub.upload_dataset(
            {"video": chunk},
            name=f"baseline_eval_part_{i // CHUNK_SIZE + 1}",
        )
        job = qai_hub.submit_inference_job(
            model=target_model,
            device=device,
            inputs=dataset,
            options="",
        )
        print(f"  Chunk {i // CHUNK_SIZE + 1}: inference job {job.job_id} → {job.url}")
        jobs.append(job)

    print("[Inference] Waiting for all jobs ...")
    combined_logits = []
    for job in jobs:
        assert job.wait().success, f"Inference job FAILED: {job.url}"
        output = job.download_output_data()
        key = "class_probs" if "class_probs" in output else list(output.keys())[0]
        combined_logits.extend(output[key])

    print(f"[Inference] {len(combined_logits)} results collected — saving to {output_h5.name}")
    with h5py.File(str(output_h5), "w") as f:
        grp = f.create_group("data/0")
        for i, arr in enumerate(combined_logits):
            arr = np.array(arr)
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            grp.create_dataset(f"batch_{i}", data=arr)


# ---------------------------------------------------------------------------
# Step 3 — Compute Top-1 / Top-5 accuracy
# ---------------------------------------------------------------------------

def topk_accuracy(preds: torch.Tensor, targets: torch.Tensor,
                  topk: tuple[int, ...] = (1, 5)) -> list[float]:
    maxk = max(topk)
    _, pred = preds.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.t().eq(targets.view(1, -1).expand_as(pred.t()))
    return [(correct[:k].reshape(-1).float().sum() / preds.size(0) * 100.0).item()
            for k in topk]


def evaluate(h5_path: Path, manifest_path: Path, class_map_path: Path) -> None:
    with open(class_map_path, encoding="utf-8") as f:
        class_to_idx: dict = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Load logits from HDF5
    logits_list = []
    with h5py.File(str(h5_path), "r") as f:
        grp  = f["data/0"]
        keys = sorted(grp.keys(), key=lambda x: int(x.split("_")[1]))
        for k in keys:
            logits_list.append(grp[k][...].squeeze())
    logits = torch.as_tensor(np.stack(logits_list, axis=0), dtype=torch.float32)
    probs  = torch.softmax(logits, dim=1)

    # Load ground-truth labels from manifest
    labels = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            lbl = json.loads(line)["label"]
            if lbl not in class_to_idx:
                raise KeyError(f"Label '{lbl}' not found in class_map.json")
            labels.append(class_to_idx[lbl])
    label_tensor = torch.tensor(labels, dtype=torch.int64)

    n_preds, n_labels = probs.shape[0], label_tensor.shape[0]
    if n_labels > n_preds:
        print(f"ℹ️  Partial results: {n_preds}/{n_labels} — truncating labels.")
        label_tensor = label_tensor[:n_preds]

    # Per-sample preview (first 20)
    pred_idx = torch.argmax(probs, dim=1)
    print("\n--- Per-sample predictions (first 20) ---")
    for i in range(min(20, pred_idx.shape[0])):
        p, g   = pred_idx[i].item(), label_tensor[i].item()
        mark   = "✓" if p == g else "✗"
        print(f"  [{i:3d}] {mark}  pred={idx_to_class[p]:<35s}  gt={idx_to_class[g]}")

    acc1, acc5 = topk_accuracy(probs, label_tensor, topk=(1, 5))
    print(f"\n{'='*52}")
    print(f"  Samples evaluated : {probs.shape[0]}")
    print(f"  Top-1 Accuracy    : {acc1:.2f}%")
    print(f"  Top-5 Accuracy    : {acc5:.2f}%")
    print(f"{'='*52}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("LPCVC Track 2 — Baseline Evaluation")
    print(f"  Data root  : {DATA_ROOT}")
    print(f"  Model      : {DLC_PATH.name}")
    print(f"  Frames     : {CLIP_LEN} @ {FRAME_RATE} fps  →  input (1,3,{T},{H},{W})")
    print(f"  Channel    : {'last (NTHWC)' if IS_CHANNEL_LAST else 'first (NCTHW)'}")
    print("=" * 60)

    assert DATA_ROOT.exists(), f"Dataset not found: {DATA_ROOT}"
    assert DLC_PATH.exists(),  (
        f"Compiled model not found: {DLC_PATH}\n"
        "  Run recompile_baseline.py first to compile and download the .bin."
    )
    assert CLASS_MAP.exists(), f"class_map.json not found: {CLASS_MAP}"

    # 1. Preprocess
    manifest_path = preprocess(DATA_ROOT, OUT_ROOT)

    # 2. Load tensors
    tensors = load_tensors(OUT_ROOT)
    print(f"[Eval] Loaded {len(tensors)} tensors, shape={tensors[0].shape}")

    # 3. On-device inference
    run_inference(DLC_PATH, tensors, OUTPUT_H5)

    # 4. Accuracy
    evaluate(OUTPUT_H5, manifest_path, CLASS_MAP)


if __name__ == "__main__":
    main()
