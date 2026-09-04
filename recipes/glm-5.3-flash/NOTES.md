# GLM-5.3-Flash on 4x RTX 5090: field notes

Companion to README.md. Measured 2026-08-27 and 2026-08-28 with
`tools/llm-bench` and the server's timing lines, greedy, single stream,
UD-IQ1_S unless stated. The GLM day, in order:

| step | shallow decode | decode @85K | prefill @3.5K / @85K |
|---|---|---|---|
| branch as published (IQ1_M, 5 GPUs) | 43 | 14.5 | 871 / 599 |
| rebuild at 2e0e57f (fused indexer, f16 masks) | 54 | 17.7 | 886 / 709 |
| switch to IQ1_S | 53-58 | 17.7 | 1,293 / 842 |
| patch 1: kpool `can_reuse` | 81 | 19.7 | unchanged |
| patch 2: NextN/MTP head, n=4 | 136 | 56.8 | unchanged |
| patch 3: KDA fused epilogues + `ssm_a` NOSCAN | 130-138 | 57.8 | 1,400 / 813 |

## The patches

1. **`glm5next: implement can_reuse for the kpool graph input`.** The DSA
   pooling input (`llm_graph_input_kpool`) never overrode `can_reuse`, so it
   inherited `return false` and the full 45-layer graph was rebuilt on the
   CPU for every decode token (`graphs reused = 0`). The override rebinds
   both cache contexts and mirrors `build_inp_kpool`'s shape arithmetic.
   Graphs reused 298/300, decode +37%, greedy output byte-identical.
2. **`glm5next: implement the NextN/MTP draft head`.** The branch had it
   stubbed ("GLM5NEXT has no MTP graph yet") although the GGUF ships
   `blk.45.nextn.*`. Ported the glm_dsa `graph_mtp` pattern: the NextN block
   is a dense MLA + MoE layer with a plain residual, so it reuses
   `build_dsa_layer` and `build_layer_ffn` in a context-only graph; the trunk
   exports the post-output-norm hidden state. Acceptance 0.90-0.98 on code.
   Speculation amortizes the DSA indexer's O(n) scan per token, which is why
   the gain is largest at depth (19.7 -> 56.8 at 86K).
3. **`glm5next: fused KDA gate epilogues + ssm_a NOSCAN placement`.** The
   fused ops themselves were worth ~1% (MTP already amortizes each graph
   over ~5 tokens). The keeper found while debugging them: `ssm_a` was
   created as `LLM_TENSOR_SSM_A`, whose placement probe tests SSM_SCAN,
   which CUDA rejects, so all 34 tensors were silently CPU-resident and
   copied to the device every graph. `LLM_TENSOR_SSM_A_NOSCAN` (MUL probe),
   as qwen3next already does. Check NOSCAN on any new hybrid architecture.
4. **`kpool: parallelize the sel/cand/pool_bias mask fill`.** Threaded the
   row loop; byte-identical output, measured 0% (kept as defensive: the
   masks were not the serial cost).
5. **`kv-cache: scale the n_kv graph-reuse pad with the ubatch size`.**
   Lets prefill chunks reuse graphs (0 -> 166/166); build cost is
   milliseconds against a ~600 ms chunk, so measured 0%. Kept for the
   graph-reuse counters it makes honest.

## Draft length

n-max 2: 103 shallow / 115 code / 42.7 @86K. n-max 4: 136 / 136 / 56.8
(default). n-max 6: 121 / 85 / 68.9: wins only copy-heavy depth, acceptance
collapses on fresh generation.

## Prefill: what did NOT work, so you do not repeat it

Raw deep prefill sits at 775-935 tok/s @86K across boots (about 7%
cross-boot variance; treat single runs accordingly). Each of these moved it
by roughly 0%: threaded mask fills, prefill graph-reuse bucketing,
`GGML_CUDA_P2P=1` (flat here, unlike DeepSeek-V4's +24%; harmless, kept),
a hunt for FA width fallbacks (not happening), and 6-GPU pipeline
parallelism (the reserve finally fits at 262K on 6 cards, prefill still
801 @86K). The host thread pegs at 100% CPU during prefill while the GPUs
average 15-20%: the serial cost is the per-ubatch graph rebuild plus the
indexer's top-k, which issues one CUB segmented sort per query row per MLA
layer (~5.6K launches per 512-token chunk). `-ub 1024` is +9% prefill for
-12% decode; the default stays 512.

## Context checkpoints: the real prefill win for agents

Hybrid-recurrent models re-prefill the full context every turn without
checkpoints (measured: 6.8 s per turn on an 8.8K prompt, every turn). With
`--ctx-checkpoints 16` (min spacing 8,192) an append-only turn costs 0.45 s
and a new session sharing the system prompt 1.07 s (it rolls back to the
nearest checkpoint inside the shared prefix). Prefixes shorter than 8,192
have no rollback point (cached=0 there is expected). `--kv-unified` BREAKS
checkpoint restore (cached=0 on every resume), so multi-slot layouts must
use split slots (`--parallel N` without `--kv-unified`).

## Memory

- ~100 KB per context token all-in at 131K with `-fa on`: MLA latent 11
  KB/t, indexer cache 5.5 KB/t, the rest compute buffers and masks.
- 4 GPUs at 262,144: 110/127 GiB placed, tightest card 1.9 GiB free. Five
  cards at 262K OOM'd at compute-buffer allocation in an earlier layout;
  the 4-card layout is a hair faster (one fewer pipeline hop).
- Six cards fit two 262,144 slots (`--parallel 2`) for a main session plus
  sub-agents: main resume after two sub-agents = cached 11,051/11,065 in
  0.91 s.

## Operational

- `GGML_CUDA_DISABLE_GRAPHS=1`: CUDA graphs never reuse on this
  architecture (the top-k's CUB pool allocations break capture); disabling
  them was +2-8%.
- Correctness watch: the competing mainline PR states `-fa off` and
  `NVIDIA_TF32_OVERRIDE=0` are required for correct output (the MLA latent
  is cast to F16 in `build_attn_mha`). We run `-fa on` to fit 131K+ and our
  spot-checks pass; do a fixed-seed on/off diff before trusting very long
  outputs. Never set `--cache-type-k/v q8_0` on this family.
- IQ1_M vs IQ1_S: decode identical (both MMVQ at decode on Blackwell),
  prefill +46% @3.5K / +19% @85K on IQ1_S because IQ1_M has no batched MMQ
  kernel. IQ1_S is the default.
