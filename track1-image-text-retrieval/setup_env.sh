#!/usr/bin/env bash
# Setup script for LPCVC Track 1 conda environment
# Tested with: CUDA 12.x driver, Python 3.10
# Key versions: PyTorch 2.4.1+cu121, qai-hub-models 0.47.0
# Usage: bash setup_env.sh

set -e

ENV_NAME="lpcvc1"
PYTHON_VERSION="3.10"
CONDA_BASE="$(conda info --base 2>/dev/null || echo $HOME/miniconda3)"
PYTHON="${CONDA_BASE}/envs/${ENV_NAME}/bin/python"
PIP="${CONDA_BASE}/envs/${ENV_NAME}/bin/pip"

echo "========================================"
echo "  LPCVC Track 1 — Environment Setup"
echo "  PyTorch 2.4.1+cu121 | qai-hub 0.45"
echo "========================================"

# Remove existing env if present
if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Removing existing '${ENV_NAME}' env..."
    conda env remove -n "${ENV_NAME}" -y
fi

echo "Creating conda env with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

echo ""
echo "--- [1/5] Upgrading pip ---"
"${PIP}" install --upgrade pip setuptools wheel

echo ""
echo "--- [2/5] PyTorch 2.4.1 + CUDA 12.1 ---"
# CUDA 12.1 wheels are forward-compatible with CUDA 12.x drivers
"${PIP}" install \
    torch==2.4.1+cu121 \
    torchvision==0.19.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "--- [3/5] Core scientific & vision packages ---"
"${PIP}" install \
    "numpy>=1.23,<2.0" \
    pandas \
    scikit-learn \
    Pillow \
    opencv-python
# Pin huggingface-hub<1.0 and transformers<5.0 — required by qai-hub-models 0.47.0
# (qai-hub-models requires huggingface-hub>=0.34,<1.0; transformers 5.x needs >=1.3.0)
"${PIP}" install "transformers>=4.40,<5.0" "huggingface-hub>=0.34,<1.0"

echo ""
echo "--- [4/5] QAI Hub + ONNX ---"
# qai-hub 0.45 + qai-hub-models 0.47 require torch>=2.4
"${PIP}" install qai-hub==0.45.0
"${PIP}" install qai-hub-models==0.47.0
"${PIP}" install \
    onnx>=1.16.0 \
    onnxruntime>=1.18.0 \
    onnxscript

echo ""
echo "--- [5/5] OpenAI CLIP ---"
"${PIP}" install "clip @ git+https://github.com/openai/CLIP.git"

echo ""
echo "--- Verification ---"
"${PYTHON}" - <<'EOF'
import torch, clip, transformers, onnx, qai_hub
print(f"  torch:           {torch.__version__}")
print(f"  torchvision:     ", end=""); import torchvision; print(torchvision.__version__)
print(f"  transformers:    {transformers.__version__}")
print(f"  onnx:            {onnx.__version__}")
print(f"  clip:            ok")
print(f"  qai_hub:         {qai_hub.__version__}")
import qai_hub_models; print(f"  qai_hub_models:  ok")
print(f"  CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "========================================"
echo "  Done! Activate with:"
echo "    conda activate ${ENV_NAME}"
echo ""
echo "  Then configure QAI Hub API key:"
echo "    qai-hub configure --api_token YOUR_TOKEN"
echo "========================================"
