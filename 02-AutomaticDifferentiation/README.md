# Week 02 — Automatic Differentiation

The lab turns on the **cost asymmetry**: for
$f:\mathbb{R}^n\to\mathbb{R}^m$, forward mode costs $O(n)$ passes and reverse mode $O(m)$.
Deep learning lives at the $m = 1$ corner, which is why backpropagation won.

- **Why not finite differences.** The truncation/roundoff vice, the
  $h \approx \sqrt{\varepsilon}$ optimum, and why float32 FD cannot beat ~$10^{-4}$
  relative error. AD is ~400x better with no $h$ to tune.
- **JVP and VJP are the only two primitives.** `jacfwd` and `jacrev` are rebuilt by hand
  from `jax.jvp` / `jax.vjp` to show columns-vs-rows.
- **The cost model, measured** (§4 — the graded deliverable): the `jacfwd`/`jacrev`
  crossover sweep. The crossover lands exactly at $n = m = 32$; the growing slopes measure
  $\approx +0.6$ rather than $+1$ because `vmap` batches the passes into one wide matmul.
  The law counts arithmetic; the hardware decides its price.
- **`grad` in practice**: `value_and_grad`, `argnums`, `has_aux`, and gradients as PyTrees
  structurally identical to the parameters — the setup for Weeks 3, 5 and 9.
- **Second order**: Rosenbrock, GD vs Newton. Quadratic convergence is shown to be a
  *local* theorem: the full step makes $f$ 58x worse before the digits start doubling,
  with a positive-definite Hessian throughout (so the step size, not the curvature, is
  the culprit).
- **HVP at $O(n)$**: forward-over-reverse costs ~1 gradient (measured flat in $n$) with
  $O(n)$ memory; Newton-CG is the matrix-free payoff.
- **custom_vjp**: fixing a softplus $NaN$ with a custom backward rule (correctness), then deliberately overriding the gradient with a straight-through estimator (wrong on purpose, but useful. We also use check_grads to verify that the custom derivative is correct when it is supposed to be.
