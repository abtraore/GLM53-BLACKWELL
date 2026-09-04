#!/bin/bash
# GLM-5.3-Flash UD-IQ1_S on 4x RTX 5090, llama.cpp (Unsloth glm5next branch + our 5 patches),
# NextN/MTP draft n=4, 262,144 context, 16 context checkpoints.
# Measured 2026-08-27: decode 130-138 tok/s shallow, 57.8 at 86K; prefill 1,400 shallow (spec off),
# 813 at 85K. Build the image first:
#   docker build -t glm53-blackwell/llamacpp:r1 .
# Knobs (see NOTES.md): SPEC_N=4 (2 -> 103/115, 6 -> 121/85 shallow but 69 @86K), CTX, GPUS.
set -e
GPUS="${GPUS:-0,1,2,3}"; PORT="${PORT:-8039}"; CTX="${CTX:-262144}"; SPEC_N="${SPEC_N:-4}"; CKPT="${CKPT:-16}"
HF="${HF_HOME:-$HOME/.cache/huggingface}"
GGUF="${GGUF:-$(ls "$HF"/hub/models--unsloth--GLM-5.3-Flash-GGUF/snapshots/*/UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00001-of-*.gguf | head -1)}"
docker run -d --restart unless-stopped --name glm-5.3-flash \
  --gpus all -e CUDA_VISIBLE_DEVICES="$GPUS" \
  -e GGML_CUDA_DISABLE_GRAPHS=1 -e GGML_CUDA_P2P=1 \
  -v "$HF":"$HF":ro -p "$PORT":8000 \
  glm53-blackwell/llamacpp:r1 \
  -m "$GGUF" --alias llamacpp/glm-5.3-flash \
  --host 0.0.0.0 --port 8000 -ngl 999 -fa on --parallel 1 --jinja \
  -c "$CTX" --ctx-checkpoints "$CKPT" \
  --spec-type draft-mtp --spec-draft-n-max "$SPEC_N" \
  --temp 1.0 --top-p 0.95 --metrics
echo "serving on http://localhost:$PORT/v1 as llamacpp/glm-5.3-flash (cold load from disk ~20 min, cache-warm ~8)"
