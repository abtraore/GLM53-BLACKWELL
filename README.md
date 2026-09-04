# GLM53-BLACKWELL

Verified GLM-5.3-Flash serving recipes for consumer Blackwell (RTX 5090,
sm120), with the llama.cpp fixes that made it usable. Every number here was
measured on real machines with the exact commands and patches published next
to it. No screenshots without flags, no "trust me" throughput.

Community project, not affiliated with Z.ai or Unsloth.

## Hardware

Current fleet: one box, "saturn": 6x RTX 5090 32 GB (sm120, PCIe, no NVLink),
AMD 5955WX, 125 GB RAM. Recipes state how many of the six cards they use.

## Recipes

Newest first: the top row is always the most recently added or updated
recipe.

| model | engine | hardware | context | decode | prefill | recipe |
|---|---|---|---|---|---|---|
| GLM-5.3-Flash (UD-IQ1_S) | llama.cpp (Unsloth glm5next branch) + 5 patches, NextN MTP n=4 | 4x 5090 | 262,144 | 130-138 tok/s (58 @86K) | 1,400 tok/s (813 @85K) | [recipe](recipes/glm-5.3-flash/) |

## What we learned on Blackwell

- **Day zero the model decoded at 43 tok/s and 14.5 at 85K depth.** Three
  patches took it to 130-138 and 58: a `can_reuse` for the DSA pool graph
  input (the whole 45-layer graph was rebuilt on the host for every token,
  +37% alone), the NextN/MTP draft head that ships inside the GGUF (the
  branch had it stubbed), and an `ssm_a` placement fix (the tensor was
  created with the SSM_SCAN probe, which CUDA rejects, so 34 tensors sat in
  host memory and were copied to the GPU every graph).
- **Prefill is host-bound on this architecture.** Five different fixes moved
  it 0%; the serial cost is the per-ubatch graph rebuild plus the indexer's
  one-CUB-sort-per-query-row top-k. `-ub 1024` is +9% prefill for -12%
  decode, so the default stays 512.
- **`--ctx-checkpoints 16` is the real prefill win for agent traffic**:
  without it a hybrid-recurrent model re-prefills the whole context every
  turn (6.8 s per turn on an 8.8K prompt); with it an append-only turn costs
  0.45 s and a new session sharing the system prompt 1.07 s.
- **IQ1_S beats IQ1_M**: identical decode (both use MMVQ on Blackwell) and
  19-46% faster prefill because IQ1_S has a batched MMQ kernel and IQ1_M
  does not. 10 GB smaller too.
- **Upstream status (2026-09-04)**: Unsloth's branch, now llama.cpp PR
  #27754, added its own NextN/MTP support on 2026-08-30. As of commit
  629b50552 it still lacks the `can_reuse` and the `ssm_a` NOSCAN fixes
  published here.

## Method

Every recipe was benchmarked with `tools/llm-bench` (stdlib only, any
OpenAI-compatible endpoint) and the server's own timing lines: warmup
discarded, unique prompts for prefill so the prefix cache cannot serve a
repeat, decode measured first token to last, single stream unless a row
says otherwise. Greedy spot-checks (arithmetic, code, reasoning_content
parsing) gate every number.
