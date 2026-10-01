# Week 03 — The JAX Ecosystem


The organizing idea is the **four states** — model parameters · optimizer state · batch statistics · PRNG — kept in four separate objects on purpose, because Week 9 makes each one a different decision.

- **A model *is* a PyTree.** `eqx.Module` is a frozen dataclass registered as a PyTree,
  so `jax.grad(loss)(model)` needs no framework machinery and the gradient of a model is
  *a model*. Static fields live in the treedef, not the leaves.
- **The filter problem, tripped deliberately.** Plain `jit` *and* `grad` both reject
  `eqx.nn.MLP` because `jax.nn.relu` sits in the tree as a non-array leaf. Fixed with
  `eqx.partition`/`combine` and `filter_jit`/`filter_grad`. Exercise 1 freezes a subtree
  via a boolean filter tree — the LoRA/PEFT mechanism of Week 15.
- **An optimizer is a gradient transformation** `(init, update)`. SGD-with-momentum is
  rebuilt from scratch and matches `optax.sgd` to **exactly 0.0**. `chain` composition
  and schedules; Exercise 2 reimplements `clip_by_global_norm` (Week 12's DP-SGD link).
- **The four states** (§3, markdown table): who mutates each, and what FL does with it —
  parameters shipped, optimizer state kept local, batch stats contested (FedBN),
  PRNG `fold_in` per client.
- **The baseline model** (§4): small Equinox ResNet, per-example `__call__` lifted by
  `filter_vmap` with `axis_name="batch"` for BatchNorm; `make_with_state` separates
  params from batch stats at birth. 9,876 params / 165 batch-stat entries.
- **Batch statistics are a real state** (§6 — the sharpest lesson): $\rho = 0.99$ vs
  $\rho = 0.9$ give **identical training loss** (0.0100) and **different test accuracy**
  (0.855 vs 1.000). The running stats are stale, not the parameters; nothing raises an
  error. This is the motivation for FedBN in Week 11.
- **Grain** (§7): composed, seeded, random-access pipelines — `len()` and `[i]` work,
  which is what Week 10's "client 7 resumes its Dirichlet shard" requires.
- **Equinox vs Flax NNX** (§8, markdown table): both must produce PyTrees before JAX will
  touch them; NNX defers the split to the boundary, Equinox never hides it. The spine
  picks Equinox because hidden state is exactly what Weeks 9–13 are about.
- **Diffrax + Lineax** (§9): `grad` differentiates *programs*, not networks — gradients
  through an adaptive ODE solver match $-e^{-a}$ to ~7 decimals, and through a CG linear
  solve to float32 precision. Exercise 4 fits an ODE parameter through the solver and
  checks the answer against a grid scan rather than assuming convergence.
