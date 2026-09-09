# High-Performance Machine Learning (DCS542)
## JAX, Distributed Training, and Federated Learning


<!-- Two threads are restated all semester:

1. **Local SGD → FedAvg.** FedAvg carries Local SGD's local-compute + periodic-averaging
   structure into the federated setting, which adds non-IID data, partial participation,
   unequal client sizes, and systems heterogeneity. 
2. **One codebase, seven evolutions.** The 11 graded labs evolve a single Equinox/Optax
   codebase from a single-device ResNet to a federated, compressed, private, defended system. -->

## Course map

| Wk | Notebook | Topic | Lab | Stage | Reading |
|----|----------|-------|-----|-------|---------|
| 01 | [Intro to HPML](01-IntroHPML/) | CPU/GPU/TPU, roofline | — | — | PPT |
<!-- | 02 | [Automatic Differentiation](02-AutomaticDifferentiation/) | Reverse vs forward AD, `custom_vjp` | 1 | — | PPT |
| 03 | [JAX Ecosystem](03-JAXEcosystem/) | Equinox · Optax · Diffrax · Lineax · Grain | 2 | 1 | PPT |
| 04 | [Efficient JAX](04-EfficientJAX/) | `vmap`/`scan`, PRNG, recompile traps, profiling | 3 | 1 | PPT |
| 05 | [Data Parallelism](05-DataParallelism/) | Ring AllReduce, `Mesh`/`NamedSharding`/`shard_map` | 4 | 2 | PPT |
| 06 | [Model Parallelism](06-ModelParallelism/) | Tensor/pipeline parallel, `remat`, ZeRO | 5 | 2 | PPT |
| 07 | [Distributed Optimization](07-DistributedOptimization/) | Local SGD → **FedAvg bridge** | 6 | 3 | Ch. 4 |
| 08 | [Foundations of FL](08-FoundationsOfFL/) | FL foundations, JAX–Flower glue| — | 4 | Ch. 2–3 |
| 09 | [FL Optimization](09-FLOptimization/) | Client drift, FedProx | 7 | 4 | Ch. 5 |
| 10 | [Communication-Efficient FL](10-CommunicationEfficientFL/) | Sampling, quantization, sparsification | 8 | 5 | Ch. 6 |
| 11 | [Heterogeneity & Personalization](11-HeterogeneityPersonalization/) | Non-IID types, graph FL | 9 | 5 | Ch. 7 |
| 12 | [Privacy](12-Privacy/) | DLG, DP-SGD, example- vs client-level DP | 10 | 6 | Ch. 9 |
| 13 | [Security](13-Security/) | Byzantine, Krum/median/trimmed mean, failure modes | 11 | 7 | Ch. 10 |
| 14 | [Trustworthy FL Systems](14-TrustworthyFLSystems/) | Regulation → measurement, EU AI Act | Position paper | — | Ch. 8 |
| 15 | [Federated Foundation Models](15-FederatedFoundationModels/) |  FedLoRA, presentations I | — | — | Papers |
| 16 | [Final Project](16-FinalProject/) | Presentations II; synthesis; final report | — | — | — | -->

Textbook: *Federated Learning: From Theory to Practice* (arXiv:2505.19183), mapped to
Ch. 2–10 across Weeks 7–14. Part I–II run on PPT + official JAX/Flower docs, since the
textbook does not cover JAX or HPC.

## The Lab Session


```
Stage 1  (W3–W4)   Single device: Equinox ResNet + Optax + jit/vmap/scan
Stage 2  (W5–W6)   Multi-device: jax.sharding (Mesh, NamedSharding, shard_map)
Stage 3  (W7)      Local SGD: convergence/communication trade-off vs period τ
Stage 4  (W8–W9)   FL transition: the same code under Flower simulation (FedAvg)
Stage 5  (W10–W11) Non-IID (Dirichlet) + compression + client sampling
Stage 6  (W12)     Privacy: DP-SGD (per-example clipping via vmap), gradient leakage
Stage 7  (W13)     Robustness: Byzantine injection → robust aggregation
```
 