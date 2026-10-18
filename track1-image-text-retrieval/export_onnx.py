import torch
import os
import clip

# --- Configuration ---
ONNX_DIR = "exported_onnx"
CLIP_MODEL = "ViT-B/32"   # 4x fewer visual patches (49) vs ViT-B/16 (196) → faster on-device
device = torch.device("cpu")
# ---------------------

os.makedirs(ONNX_DIR, exist_ok=True)
print(f"Saving ONNX files to: {os.path.abspath(ONNX_DIR)}")
print(f"Loading CLIP model: {CLIP_MODEL}")

# --- Dummy inputs (same shapes as evaluation pipeline) ---
DUMMY_IMAGE_INPUT = torch.rand(1, 3, 224, 224, dtype=torch.float32, device=device)
DUMMY_TEXT_INPUT = torch.randint(0, 49408, (1, 77), dtype=torch.int64, device=device)

# --- Load CLIP ViT-B/32 ---
clip_model, _ = clip.load(CLIP_MODEL, device=device)
clip_model = clip_model.to(torch.float32).eval()

print(f"Loaded {CLIP_MODEL} — visual patches: {(224 // 32)**2} (vs 196 for ViT-B/16)")


# -------------------------------------------------------
# Image Encoder
# -------------------------------------------------------
class ImageEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.visual = clip_model.visual

    def forward(self, images):
        return self.visual(images)


# -------------------------------------------------------
# Text Encoder
# -------------------------------------------------------
class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, token_ids):
        x = self.token_embedding(token_ids)          # [B, 77, 512]
        x = x + self.positional_embedding             # [B, 77, 512]
        x = x.permute(1, 0, 2)                        # [77, B, 512] (transformer expects seq-first)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)                        # [B, 77, 512]
        x = self.ln_final(x)
        eos_index = token_ids.argmax(dim=-1)           # position of EOT token
        x = x[torch.arange(x.shape[0]), eos_index]   # [B, 512] — take EOS token features
        x = x @ self.text_projection                   # [B, 512]
        return x


# --- Instantiate wrappers ---
image_encoder = ImageEncoderWrapper(clip_model).eval()
text_encoder = TextEncoderWrapper(clip_model).eval()


# -------------------------------------------------------
# Export Image Encoder
# -------------------------------------------------------
image_onnx_path = os.path.join(ONNX_DIR, "image_encoder.onnx")
print(f"\nExporting Image Encoder → {image_onnx_path}")

torch.onnx.export(
    image_encoder,
    DUMMY_IMAGE_INPUT,
    image_onnx_path,
    input_names=["image"],
    output_names=["embedding"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes=None,
    verbose=False,
    export_params=True,
    training=torch.onnx.TrainingMode.EVAL,
)
print("Image Encoder exported ✓")


# -------------------------------------------------------
# Export Text Encoder
# -------------------------------------------------------
text_onnx_path = os.path.join(ONNX_DIR, "text_encoder.onnx")
print(f"\nExporting Text Encoder  → {text_onnx_path}")

torch.onnx.export(
    text_encoder,
    DUMMY_TEXT_INPUT,
    text_onnx_path,
    input_names=["text"],
    output_names=["text_embedding"],
    opset_version=17,
    do_constant_folding=True,
    dynamic_axes=None,
    verbose=False,
    export_params=True,
    training=torch.onnx.TrainingMode.EVAL,
)
print("Text Encoder exported ✓")

print("\nExport complete.")
print(f"  Image encoder: {image_onnx_path}")
print(f"  Text encoder:  {text_onnx_path}")
