#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="$(cd -- "$EXAMPLE_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
PEFT_DIR="${PEFT_DIR:-$BUNDLE_DIR/weights/vlm}"
DABDETR_DIR="${DABDETR_DIR:-$BUNDLE_DIR/weights/detector}"
OUTPUT_DIR="${OUTPUT_DIR:-$BUNDLE_DIR/results}"
LM_BASE_ID="${LM_BASE_ID:-Qwen/Qwen3-VL-8B-Instruct}"

exec "$PYTHON_BIN" "$BUNDLE_DIR/inference.py" \
  --test-json "$EXAMPLE_DIR/input.json" \
  --image-root "$EXAMPLE_DIR/images" \
  --peft-dir "$PEFT_DIR" \
  --dabdetr-dir "$DABDETR_DIR" \
  --lm-base-id "$LM_BASE_ID" \
  --output-dir "$OUTPUT_DIR" \
  --max-new-tokens 1024 \
  --score-threshold 0.001 \
  --max-det 300 \
  --progress-every 1
