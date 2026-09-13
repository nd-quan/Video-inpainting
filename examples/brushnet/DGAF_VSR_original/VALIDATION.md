# Validation — 2026-09-11

Environment: `/home/cilab/ndquan/envs/guided_diff/bin/python`, Torch 2.0.1 CUDA,
Accelerate 0.21. Tests used the configured real baseline and frozen RAFT student.

- 14 CPU tests passed: geometry, confidence, source/target flow orientation,
  independent sequence batches, same-timestep recurrence, forward/backward order,
  per-sweep baseline reset, CFG, repeated pipeline calls, sequence merging/gaps,
  prediction conventions, native model round-trip and temporal gradients.
- Preflight: 4,870 training triplets, T3/S1.
- Full-sequence indexing: 18 contiguous sequences, 4,906/4,906 source frames;
  no overlap duplication or missing tail frames in the configured train dataset.
- Direct/no confidence, T3/128, one GPU, one optimizer update:
  loss 0.3232937; new temporal input gradient norm 0.2317266.
- DGAF/confidence, T3/128, two GPUs, accumulation 2, one optimizer update:
  loss 0.2909715; temporal input gradient norm 0.1699794.
- Direct/no confidence, T3/512, one GPU, one optimizer update:
  loss 0.2506034; temporal input gradient norm 0.0297038.
- Real DGAF/no-confidence inference: 3 frames, 128 resolution, 3 DDIM steps,
  CFG 7.5; finite latent output and decoded PNGs. This exercises all three
  forward/backward/forward sweeps and both frozen/temporal BrushNet paths.
- Python compilation, shell syntax and tracked diff whitespace checks passed.

Smoke artifacts are in `/tmp/dgaf_original_direct_smoke_20260911`,
`/tmp/dgaf_original_ddp_smoke_20260911`, `/tmp/dgaf_original_512_smoke_20260911`,
and `/tmp/dgaf_original_inference_smoke_20260911`.

The 128 training checks preceded replacing the algebraically equivalent
effective-noise expression with the direct sampled-noise/velocity target;
the 512 check and final CPU suite used the final target implementation.

No long training, full-length sequence quality evaluation, or new-version
checkpoint resume round-trip was run. The checkpoint machinery is inherited
from the previous implementation; schema 2 prevents mixing training protocols.
Full-sequence indexing and short real inference were tested separately.
These smoke losses are functionality checks, not evidence of restoration quality.
