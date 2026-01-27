#!/bin/bash
# Download ONNX models (SuperPoint + LightGlue) and ONNX Runtime SDK
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ─── ONNX Models ─────────────────────────────────────────────────────────────
MODELS_DIR="$PROJECT_DIR/models"
mkdir -p "$MODELS_DIR"

# v0.1.0 has separate SuperPoint + LightGlue models (not fused pipeline)
LIGHTGLUE_ONNX_BASE="https://github.com/fabio-sim/LightGlue-ONNX/releases/download/v0.1.0"

if [ ! -f "$MODELS_DIR/superpoint.onnx" ] || [ "$(wc -c < "$MODELS_DIR/superpoint.onnx")" -lt 1000 ]; then
    echo "Downloading superpoint.onnx..."
    curl -L "$LIGHTGLUE_ONNX_BASE/superpoint.onnx" -o "$MODELS_DIR/superpoint.onnx"
else
    echo "superpoint.onnx already exists, skipping."
fi

if [ ! -f "$MODELS_DIR/superpoint_lightglue.onnx" ] || [ "$(wc -c < "$MODELS_DIR/superpoint_lightglue.onnx")" -lt 1000 ]; then
    echo "Downloading superpoint_lightglue.onnx..."
    curl -L "$LIGHTGLUE_ONNX_BASE/superpoint_lightglue.onnx" -o "$MODELS_DIR/superpoint_lightglue.onnx"
else
    echo "superpoint_lightglue.onnx already exists, skipping."
fi

# ─── ONNX Runtime SDK ────────────────────────────────────────────────────────
ORT_VERSION="1.17.1"
ORT_DIR="$PROJECT_DIR/onnxruntime"

# Use --gpu flag to download the GPU variant
USE_GPU=false
for arg in "$@"; do
    if [ "$arg" = "--gpu" ]; then
        USE_GPU=true
    fi
done

if [ -d "$ORT_DIR" ]; then
    echo "ONNX Runtime SDK already exists at $ORT_DIR, skipping."
    echo "  (Delete $ORT_DIR to re-download, or use --gpu for GPU variant)"
else
    # Detect platform
    ARCH="$(uname -m)"
    OS="$(uname -s)"

    GPU_SUFFIX=""
    if [ "$USE_GPU" = true ] && [ "$OS" = "Linux" ] && [ "$ARCH" = "x86_64" ]; then
        GPU_SUFFIX="-gpu"
        echo "GPU variant requested — downloading with CUDA support"
    fi

    if [ "$OS" = "Linux" ]; then
        if [ "$ARCH" = "x86_64" ]; then
            ORT_FILENAME="onnxruntime-linux-x64${GPU_SUFFIX}-${ORT_VERSION}"
        elif [ "$ARCH" = "aarch64" ]; then
            ORT_FILENAME="onnxruntime-linux-aarch64-${ORT_VERSION}"
        else
            echo "Unsupported Linux architecture: $ARCH"
            exit 1
        fi
    elif [ "$OS" = "Darwin" ]; then
        if [ "$ARCH" = "arm64" ]; then
            ORT_FILENAME="onnxruntime-osx-arm64-${ORT_VERSION}"
        elif [ "$ARCH" = "x86_64" ]; then
            ORT_FILENAME="onnxruntime-osx-x86_64-${ORT_VERSION}"
        else
            echo "Unsupported macOS architecture: $ARCH"
            exit 1
        fi
    else
        echo "Unsupported OS: $OS"
        exit 1
    fi

    ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_FILENAME}.tgz"
    echo "Downloading ONNX Runtime ${ORT_VERSION} for ${OS}-${ARCH}${GPU_SUFFIX:+ (GPU)}..."
    curl -L "$ORT_URL" -o "/tmp/${ORT_FILENAME}.tgz"
    echo "Extracting..."
    tar -xzf "/tmp/${ORT_FILENAME}.tgz" -C "$PROJECT_DIR"
    mv "$PROJECT_DIR/$ORT_FILENAME" "$ORT_DIR"
    rm -f "/tmp/${ORT_FILENAME}.tgz"
fi

echo ""
echo "Done! Models in: $MODELS_DIR"
echo "ONNX Runtime in: $ORT_DIR"
echo ""
echo "Next steps:"
echo "  cd build && cmake .. && make -j\$(nproc)"
