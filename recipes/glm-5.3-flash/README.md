# GLM-5.3-Flash on 4x RTX 5090 (llama.cpp)

The 320B/18B-active GLM-5.3-Flash (34 KDA linear-attention layers + 11 MLA
layers with the DSA sparse indexer, 288 experts) as Unsloth's UD-IQ1_S GGUF
on four 5090s at a 262,144-token context: **130-138 tok/s** shallow decode
and **58 tok/s at 86K depth** with the NextN/MTP head that ships inside the
GGUF (acceptance 0.90-0.98 on code, mean draft length 4.6-4.9 of 5), prefill
**1,400 tok/s** shallow and **813 at 85K**. Day zero, on the branch as
published, the same setup decoded at 43 tok/s and 14.5 at depth.

Gates: greedy arithmetic, code, `reasoning_content` parsing through
`--jinja`, greedy output byte-identical to the unpatched build with
speculation off (spec-on output differs benignly: batched verification flips
greedy near-ties).

The five patches in `patches/` are what closes the gap; they apply with
`git am` on Unsloth's `glm5next/upstream` at commit 2e0e57f (llama.cpp PR
#27754). As of 2026-09-04 that PR carries its own NextN/MTP support but not
the graph-reuse and `ssm_a` fixes.

## Contents

- `Dockerfile` + `patches/`: the fork at the pinned commit plus the five
  patches, each with its upstream thread and retirement condition in the
  header
- `launch.sh`: parameterized launcher (`GPUS`/`PORT`/`CTX`/`SPEC_N`/`CKPT`)
- `NOTES.md`: what each patch fixes and how much, the knob sweeps, the
  prefill investigation, and the operational traps

## Pull the weights

~88 GB (IQ1_S, 3 shards). IQ1_M is 10 GB larger and no better (see NOTES).

```bash
hf download unsloth/GLM-5.3-Flash-GGUF --include "UD-IQ1_S/*"
```

## Start the server

```bash
git clone https://github.com/abtraore/GLM53-BLACKWELL && cd GLM53-BLACKWELL/recipes/glm-5.3-flash
docker build -t glm53-blackwell/llamacpp:r1 .
./launch.sh
```

Or the full command without the script:

```bash
HF=$HOME/.cache/huggingface
GGUF=$(ls $HF/hub/models--unsloth--GLM-5.3-Flash-GGUF/snapshots/*/UD-IQ1_S/GLM-5.3-Flash-UD-IQ1_S-00001-of-*.gguf | head -1)
docker run -d --restart unless-stopped --name glm-5.3-flash \
  --gpus all -e CUDA_VISIBLE_DEVICES=0,1,2,3 -e GGML_CUDA_DISABLE_GRAPHS=1 -e GGML_CUDA_P2P=1 \
  -v $HF:$HF:ro -p 8039:8000 \
  glm53-blackwell/llamacpp:r1 \
  -m $GGUF --alias llamacpp/glm-5.3-flash \
  --host 0.0.0.0 --port 8000 -ngl 999 -fa on --parallel 1 --jinja \
  -c 262144 --ctx-checkpoints 16 \
  --spec-type draft-mtp --spec-draft-n-max 4 \
  --temp 1.0 --top-p 0.95 --metrics
```

Serving on `http://localhost:8039/v1`, model name `llamacpp/glm-5.3-flash`.
First-use warning: a cold load of 88 GB from a SATA disk takes about 20
minutes (8 with the files in page cache); the four cards end up at
110/127 GiB with the tightest one 1.9 GiB free at 262K.
