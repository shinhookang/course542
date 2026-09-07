# Week 01 — Introduction to High-Performance Machine Learning

Reading: PPT · No graded lab (Lab 1 is Week 2)

Notebook: `Week01-IntroHPML.ipynb` (built by `_build/build_week01.py`)

The course opens by building the measuring instruments everything else depends on. No
model is trained; instead we find out what one machine can actually do, and establish the
roofline model as the tool that predicts which limit binds.

- **Why HPML is arithmetic, not vibes**: `C ≈ 6·N_params·N_tokens` prices modern models at
  tens to hundreds of device-years, making parallelism the only way training finishes.
- **The memory hierarchy, measured**: a random-gather latency sweep exposes the staircase
  from cache-resident (~1 ns/access) to DRAM-resident (~12 ns) — about a 10× miss penalty.
  Streaming bandwidth β is measured separately at DRAM-resident sizes.
- **Arithmetic intensity and the roofline**: `P(I) = min(π, I·β)`, ridge `I* = π/β`. Both
  roofs are measured on the machine, then real kernels are placed on the plot: streaming
  kernels hug the slant, matmul walks across the ridge as `I = n/6` and saturates, and the
  tiniest matmuls fall below *both* roofs — the model's domain of validity has an edge.
- **CPU vs GPU vs TPU**: latency vs throughput vs systolic dataflow. V100→H100 grew compute
  ~63× but bandwidth only ~3.7×, so the ridge climbed ~17× — new accelerators are *harder*
  to keep busy.
- **NumPy → JAX**: the four rule changes — immutability (what makes `jit`/`grad`/`vmap`/
  sharding composable), float32 by default, async dispatch (**always `block_until_ready()`**),
  and explicit PRNG keys.
- **`jit` as a roofline intervention**: fusion turns k passes over memory into one,
  multiplying intensity by k. The measured speedup rises to ~25×, then *declines* as fusion
  pushes the kernel past the ridge and compute takes over.

**Benchmarking discipline** is introduced here and enforced all semester: warm up, call
`block_until_ready()`, take a median, report a rate. The lab shows a "3000× speedup" that
is purely a forgotten `block_until_ready()`.

Exercises: 5-point stencil on the roofline (and why NumPy's temporaries miss it while
`jit` recovers most of the gap); porting a mutating/global-RNG NumPy function to JAX;
shape-triggered recompilation; placing a transformer FFN on the roofline (why decoding at
T=1 is hopelessly memory-bound, and why a d=128 net can never saturate an H100).

Threads planted for later weeks: `vmap` → Week 12 DP-SGD · recompilation → Week 4 ·
large batches vs generalization → Week 7 · small clients in the memory-bound corner →
Week 9 · compute-vs-communication ratios → Weeks 5–10.
