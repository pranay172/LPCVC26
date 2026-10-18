"""
LPCVC Track 2 — Correct Baseline Recompilation Script
======================================================
This script replaces my first attempt (which compiled through the
qai_hub_models wrapper) by building the model CORRECTLY.

Root cause of the previous failure:
  `qai_hub_models Model.from_pretrained()` exposes `model.model` as an
  attribute, but its `forward()` always calls the original Kinetics-400
  backbone via `self.model`. Patching `model.model.fc` and loading our
  checkpoint had no effect on what was actually traced and compiled.

Fix:
  Build a standalone nn.Module (NormalizedR2Plus1D) that:
    1. Applies Kinetics mean/std normalization explicitly (so it's baked in)
    2. Runs our fine-tuned 92-class r2plus1d_18 backbone
  Export this module to ONNX, then compile via QAI Hub.

Usage:
    cd track2-video-action-recognition
    python recompile_baseline.py
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import qai_hub as hub
from torchvision.models.video import r2plus1d_18

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = Path(__file__).parent / "model_checkpoints" / "model_29.pth"
NUM_CLASSES     = 92
NUM_FRAMES      = 8
DEVICE_NAME     = "Dragonwing IQ-9075 EVK"
OUTPUT_DIR      = Path(__file__).parent / "export_assets"
ONNX_PATH       = OUTPUT_DIR / "resnet_2plus1d_baseline_ep29.onnx"
SHARE_EMAIL     = "lowpowervision@gmail.com"

# Kinetics normalization stats (same as used during training)
NORM_MEAN = (0.43216, 0.394666, 0.37645)
NORM_STD  = (0.22803, 0.22145, 0.216989)

# ---------------------------------------------------------------------------
# Standalone model: normalization + fine-tuned backbone
# ---------------------------------------------------------------------------

class NormalizedR2Plus1D(nn.Module):
    """
    r2plus1d_18 with Kinetics normalization baked in.
    Input:  (N, 3, T, H, W) float32 in [0, 1]
    Output: (N, num_classes) logits
    """
    def __init__(self, num_classes: int):
        super().__init__()
        # Register normalization constants as buffers (non-trainable, exported to ONNX)
        mean = torch.tensor(NORM_MEAN).view(1, 3, 1, 1, 1)  # (1,C,1,1,1)
        std  = torch.tensor(NORM_STD ).view(1, 3, 1, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)
        # Backbone
        backbone = r2plus1d_18()
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.backbone = backbone

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # video: (N, 3, T, H, W) in [0, 1]
        video = (video - self.mean) / self.std
        return self.backbone(video)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

print("=" * 60)
print("LPCVC Track 2 — Recompiling Baseline (correct path)")
print(f"  Checkpoint : {CHECKPOINT_PATH}")
print(f"  num_classes: {NUM_CLASSES},  num_frames: {NUM_FRAMES}")
print(f"  Device     : {DEVICE_NAME}")
print("=" * 60)

assert CHECKPOINT_PATH.exists(), f"Checkpoint not found: {CHECKPOINT_PATH}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Build and load weights
print("\n[1/4] Building model and loading checkpoint ...")
model = NormalizedR2Plus1D(NUM_CLASSES)
ckpt  = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
state = ckpt["model"] if "model" in ckpt else ckpt
model.backbone.load_state_dict(state, strict=True)
model.eval()
print(f"  Loaded epoch {ckpt.get('epoch', '?')}  fc: {model.backbone.fc}")

# Sanity-check: forward pass and argmax on a random tensor
with torch.no_grad():
    dummy = torch.rand(1, 3, NUM_FRAMES, 112, 112)
    out   = model(dummy)
assert out.shape == (1, NUM_CLASSES), f"Unexpected output shape: {out.shape}"
print(f"  Sanity check passed — output shape: {out.shape}")

# 2. Export to ONNX
print(f"\n[2/4] Exporting to ONNX → {ONNX_PATH} ...")
dummy_input = torch.rand(1, 3, NUM_FRAMES, 112, 112)
torch.onnx.export(
    model,
    dummy_input,
    str(ONNX_PATH),
    input_names=["video"],
    output_names=["class_logits"],
    dynamic_axes={"video": {0: "batch"}},
    opset_version=17,
    do_constant_folding=True,
)
print(f"  ONNX saved: {ONNX_PATH.stat().st_size / 1024 / 1024:.1f} MB")

# Quick ONNX validation
import onnx
onnx.checker.check_model(str(ONNX_PATH))
print("  ONNX model is valid ✓")

# 3. Compile on AI Hub
print(f"\n[3/4] Submitting compile job to AI Hub ...")
devices = hub.get_devices(name=DEVICE_NAME)
assert devices, f"Device '{DEVICE_NAME}' not found"
device = devices[-1]

input_specs = {"video": ((1, 3, NUM_FRAMES, 112, 112), "float32")}

compile_job = hub.submit_compile_job(
    model=str(ONNX_PATH),
    device=device,
    name=f"resnet_2plus1d_baseline_ep{ckpt.get('epoch', 29)}_onnx",
    input_specs=input_specs,
    options="--target_runtime qnn_context_binary",
)
print(f"  Compile job submitted: {compile_job.url}")
print("  Waiting for compile job (may take several minutes) ...")
assert compile_job.wait().success, f"Compile FAILED: {compile_job.url}"
print("  ✓ Compile succeeded!")

# 4. Download compiled .bin
print(f"\n[4/4] Downloading compiled .bin ...")
target  = compile_job.get_target_model()
bin_path = target.download(str(OUTPUT_DIR / f"resnet_2plus1d_baseline_ep{ckpt.get('epoch',29)}_onnx.bin"))
print(f"  ✓ Downloaded: {bin_path}")

# Share with organizers
compile_job.modify_sharing(add_emails=[SHARE_EMAIL])
print(f"  ✓ Shared with {SHARE_EMAIL}")

print("\n" + "=" * 60)
print("RECOMPILATION COMPLETE")
print(f"  Compile Job ID  : {compile_job.job_id}")
print(f"  Compile Job URL : {compile_job.url}")
print(f"  Compiled model  : {bin_path}")
print()
print("Update evaluate_baseline.py DLC_PATH to:")
print(f"  DLC_PATH = TRACK2_ROOT / 'export_assets' / '{Path(bin_path).name}'")
print()
print("IMPORTANT: Update your competition submission form with the new")
print(f"  Compile Job ID: {compile_job.job_id}")
print("=" * 60)
