# High-Performance Machine Learning (DCS542)
## JAX, Distributed Training, and Federated Learning

Graduate, 16 weeks. The authoritative course design is **[`hpml-curriculum.md`](hpml-curriculum.md)**;
the build workflow and authoring rules are in **[`CLAUDE.md`](CLAUDE.md)**.

The semester is one narrative, not a topic list:

> **Machine Learning** → **Automatic Differentiation** (the engine of learning)
> → **JAX** (compiling and parallelizing AD) → **Distributed Optimization** (past one GPU)
> → **Federated Learning** (when data cannot be centralized)
> → **Trustworthy AI & Federated Foundation Models** (can we trust distributed learning?)

Two threads are restated all semester:

1. **Local SGD → FedAvg.** FedAvg carries Local SGD's local-compute + periodic-averaging
   structure into the federated setting, which adds non-IID data, partial participation,
   unequal client sizes, and systems heterogeneity. **Week 7 is the hinge.**
2. **One codebase, seven evolutions.** The 11 graded labs evolve a single Equinox/Optax
   codebase from a single-device ResNet to a federated, compressed, private, defended system.

## Course map

| Wk | Notebook | Topic | Lab | Stage | Reading |
|----|----------|-------|-----|-------|---------|
| 01 | [Intro to HPML](01-IntroHPML/) | CPU/GPU/TPU; roofline | — | — | PPT |
| 02 | [Automatic Differentiation](02-AutomaticDifferentiation/) | Reverse vs forward AD; HVP; `custom_vjp` | 1 | — | PPT |
| 03 | [JAX Ecosystem](03-JAXEcosystem/) | Equinox · Optax · Diffrax · Lineax · Grain | 2 | 1 | PPT |
| 04 | [Efficient JAX](04-EfficientJAX/) | `vmap`/`scan`; PRNG; recompile traps; profiling | 3 | 1 | PPT |
| 05 | [Data Parallelism](05-DataParallelism/) | Ring AllReduce; `Mesh`/`NamedSharding`/`shard_map` | 4 | 2 | PPT |
| 06 | [Model Parallelism](06-ModelParallelism/) | Tensor/pipeline parallel; `remat`; ZeRO | 5 | 2 | PPT |
| 07 | [Distributed Optimization](07-DistributedOptimization/) | Local SGD → **FedAvg bridge** | 6 | 3 | Ch. 4 |
| 08 | [Foundations of FL](08-FoundationsOfFL/) | FL foundations; JAX–Flower glue; proposal talks | — | 4 | Ch. 2–3 |
| 09 | [FL Optimization](09-FLOptimization/) | Client drift; FedProx; SCAFFOLD; consensus | 7 | 4 | Ch. 5 |
| 10 | [Communication-Efficient FL](10-CommunicationEfficientFL/) | Sampling; quantization; sparsification | 8 | 5 | Ch. 6 |
| 11 | [Heterogeneity & Personalization](11-HeterogeneityPersonalization/) | Non-IID types; Ditto/FedBN/MOON; graph FL | 9 | 5 | Ch. 7 |
| 12 | [Privacy](12-Privacy/) | DLG; DP-SGD; example- vs client-level DP; SecAgg | 10 | 6 | Ch. 9 |
| 13 | [Security](13-Security/) | Byzantine; Krum/median/trimmed mean; failure modes | 11 | 7 | Ch. 10 |
| 14 | [Trustworthy FL Systems](14-TrustworthyFLSystems/) | Regulation → measurement; EU AI Act | Position paper | — | Ch. 8 |
| 15 | [Federated Foundation Models](15-FederatedFoundationModels/) | PEFT/LoRA in FL; FedLoRA; presentations I | — | — | Papers |
| 16 | [Final Project](16-FinalProject/) | Presentations II; synthesis; final report | — | — | — |

**Week ≠ Lab number.** Weeks 1 and 8 have no lab, Week 14 is a position paper, and
Weeks 15–16 are presentations. Week 2 = Lab 1, Week 3 = Lab 2, …, Week 13 = Lab 11.

Textbook: *Federated Learning: From Theory to Practice* (arXiv:2505.19183), mapped to
Ch. 2–10 across Weeks 7–14. Part I–II run on PPT + official JAX/Flower docs, since the
textbook does not cover JAX or HPC.

## The Lab Spine

The labs are not independent exercises; they are one codebase evolving.

```
Stage 1  (W3–W4)   Single device: Equinox ResNet + Optax + jit/vmap/scan
Stage 2  (W5–W6)   Multi-device: jax.sharding (Mesh, NamedSharding, shard_map)
Stage 3  (W7)      Local SGD: convergence/communication trade-off vs period τ
Stage 4  (W8–W9)   FL transition: the same code under Flower simulation (FedAvg)
Stage 5  (W10–W11) Non-IID (Dirichlet) + compression + client sampling
Stage 6  (W12)     Privacy: DP-SGD (per-example clipping via vmap), gradient leakage
Stage 7  (W13)     Robustness: Byzantine injection → robust aggregation
```

Four pieces of state stay **explicitly separated** from Week 3 onward: **model parameters ·
optimizer state · batch statistics · PRNG state**. That separation is what makes the
Stage 4 FL transition tractable.

## Notebooks are generated, not hand-edited

Every notebook is authored by a build script in [`_build/`](_build/) that assembles cells and
then **executes the whole notebook** with nbclient. A notebook that has not run top-to-bottom
is not trusted. **Edit the build script and rerun it — never edit the `.ipynb`.**

```bash
cd _build
../.venv/bin/python build_week07.py        # rebuild one week

for f in build_week[0-9]*.py; do ../.venv/bin/python "$f" || echo "FAILED: $f"; done   # rebuild all
```

Use the project venv (`.venv/bin/python`, Python 3.11); notebooks execute against the `hpml`
kernelspec. See [`CLAUDE.md`](CLAUDE.md) for the interpreter, the verified Flower recipe, and
the authoring pitfalls — and [`_build/STYLE_SPEC.md`](_build/STYLE_SPEC.md) for the house style.

## Build status

All notebooks execute end-to-end with **zero errors**.

| Week | Cells | Code | Figures | Status |
|------|------:|-----:|--------:|--------|
| 01 Intro to HPML | 57 | 26 | 5 | ✅ |
| 02 Automatic Differentiation | 62 | 32 | 5 | ✅ |
| 03 JAX Ecosystem | 68 | 29 | 6 | ✅ |
| 04 Efficient JAX | 74 | 32 | 3 | ✅ |
| 05 Data Parallelism | 65 | 31 | 5 | ✅ |
| 06 Model Parallelism | 63 | 31 | 8 | ✅ |
| 07 Distributed Optimization | 65 | 31 | 8 | ✅ |
| 08 Foundations of FL | 64 | 27 | 6 | ✅ |
| 09 FL Optimization | 63 | 36 | 7 | ✅ |
| 10 Communication-Efficient FL | 71 | 41 | 8 | ✅ |
| 11 Heterogeneity & Personalization | 61 | 33 | 9 | ✅ |
| 12 Privacy | 85 | 38 | 10 | ✅ |
| 13 Security | 60 | 34 | 7 | ✅ |
| 14 Trustworthy FL Systems | 68 | 37 | 7 | ✅ |
| 15 Federated Foundation Models | 77 | 41 | 5 | ✅ |
| 16 Final Project (synthesis) | 71 | 28 | 4 | ✅ |

**Totals: 16 notebooks · 1,074 cells · 527 code cells · 103 figures · 0 errors.**

## This machine is CPU-only — read the numbers accordingly

`jax.devices()` returns a single `CpuDevice`; the department GPU nodes are where students run
the real thing. Multi-device weeks fake devices via
`XLA_FLAGS=--xla_force_host_platform_device_count=8` **before** importing JAX: the sharding
*semantics* are real, the *performance* numbers are not. No notebook presents a CPU-simulated
device count as a scaling result. There is no network in the harness, so the notebooks use a
synthetic CIFAR-shaped dataset; the real labs point at CIFAR-10 via Grain on the GPU node.

Consequently each notebook distinguishes what it **confirmed** from what it **could not**.
Confirmed here: the pipeline bubble $(P-1)/(M+P-1)$ exact on all 28 (P,M) pairs (W6); ZeRO 8×
bit-identical (W6); Ring AllReduce's exact byte count (W5); FedAvg ≡ Local SGD to `0.000e+00`
when the federated constraints degenerate (W7); FedAvg's drift fixed point to 1e-15 (W9); the
DP accountant against Abadi et al., ε = 2.21 vs ≈2 (W12). Not confirmable here: real scaling
curves, peak device memory (XLA:CPU excludes temporaries), Stich's τ threshold, real LLM scale.

## Assessment

| Item | Weight |
|------|-------:|
| Assignment — 11 cumulative technical labs + 1 position paper | 36% |
| Project Proposal — Week 8 lightning talk + document | 20% |
| Final Project — Week 15–16 presentation + report (code release mandatory) | 40% |
| Attendance | 4% |
