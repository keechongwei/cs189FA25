# HW3 — Notebook Walkthrough (Autodiff Engine & Optimizers)

Notebook covered: `hw3.ipynb`

**Goal in the top-level README:** "modern ML tooling and datasets", training/evaluation/visualisation pipelines, comparing models under different objectives, generalisation and evaluation rigour (train/validation splits, metrics, overfitting).

> **Mismatch with the README.** The notebook is titled *"Optimizers and Backpropagation"*. It is **not** a pipeline built from modern tooling. It implements a small **automatic-differentiation library from scratch in NumPy** (`BearTensor`) plus three optimizers (SGD, Momentum, Adam), then uses them to fit a small network on a wine-quality dataset. Evaluation uses a single train/test split and MSE. The README's description fits the *spirit* (understanding optimisation and generalisation) but not the content.

---

## Libraries used

| Library | Used for |
|---|---|
| `numpy` | all tensor maths. The autograd engine wraps `ndarray` |
| `dataclasses`, `collections.deque`, `re` | graph node classes, BFS reset, string handling for graph drawing |
| `matplotlib` | optimizer-comparison plot |
| `typst` (Python package) | rendering computation-graph visualisations (provided helper code) |
| `scikit-learn` | `fetch_openml("wine-quality-red")`, `StandardScaler`, `train_test_split` |
| `otter-grader` | autograder checks |

No PyTorch is used. The point is to rebuild what `torch.autograd` does.

---

## Q1 — Building the computation graph (`BearTensor`)

Three small classes:
- `BearGrad(fn, op_str)`: stores a function that maps the **upstream gradient** to the **downstream gradient** (the local chain-rule step).
- `BearParent(parent, grad)`: pairs a parent tensor with its `BearGrad`.
- `BearTensor(name, value, parents)`: a graph node with a NumPy `value`, a list of `parents`, and an `adjoint` (the gradient).

Every operator builds a new `BearTensor` whose parents record how to push gradients back:

| Op | Local gradient (what each `BearGrad.fn` returns) |
|---|---|
| `a + b` | `g` to each parent (after unbroadcasting) |
| `a - b` | `g` to `a`, `-g` to `b` |
| `a * b` | `g * b.value` to `a`, `g * a.value` to `b` |
| `a ** p` | `g * p * a^(p-1)` |
| `a @ b` | `g @ bᵀ` to `a`, `aᵀ @ g` to `b` |
| `dot`, `sum`, `mean` | broadcast `g` (scaled by `1/D` for `mean`) |
| `relu` | `g * (x > 0)` |
| `sigmoid` | `g * s * (1 − s)` where `s = σ(x)` |

`_unbroadcast(grad, target_shape)` is the key helper: it sums away dimensions that NumPy broadcasting added so that a gradient always matches its tensor's shape. The values captured in each lambda (`y=other.value`, `s=s_shape`, …) are bound as default arguments so each closure keeps its own copy.

## Q2 — Backpropagation

- `topological_sort(node)`: recursive DFS that appends a node after all its parents, so reversing the list gives an order where a node's gradient is complete before it is pushed on.
- `reset_children`: BFS (`deque`) that zeroes every node's `adjoint` so repeated calls to `backward()` don't accumulate stale gradients.
- `backward()`: reset → topologically sort → seed `self.adjoint = 1.0` → walk the reversed order, and for each parent do `parent.adjoint += parent_grad.fn(node.adjoint)`.

These are attached to `BearTensor` after definition (`BearTensor.backward = backward`).

## Q3 — Optimizers

All three share a base class with `zero_grad()` and `step()`.
- **SGD:** `w ← w − lr · grad`.
- **Momentum:** `v ← β·v + grad`, `w ← w − lr · v` (β = 0.9; the exponential-average variant is left commented out).
- **Adam:** first and second moment estimates (β₁ = 0.9, β₂ = 0.999), **bias-corrected** `m̂ = m/(1−β₁ᵗ)`, `v̂ = v/(1−β₂ᵗ)`, update `w ← w − lr · m̂ / (√v̂ + ε)`.

**Optimizer comparison.** `compare_optimizers` trains a 24→12→1 sigmoid network on random data with identical initialisation for 100 steps at `lr = 0.05`. The resulting plot shows **SGD and Momentum diverging** (the loss explodes to around 1e301, with NumPy overflow warnings in `exp` and `square`), while **Adam stays stable**. This illustrates Adam's per-parameter normalisation helping at a learning rate that is too large for the plain methods.

## Q4 — Training a model on wine quality

- **Data:** red wine quality (1,599 samples, 11 chemical features, target 0–10), standardised with `StandardScaler`, split 60/40 (`random_state=42`).
- **Weights:** He initialisation (`std = √(2/fan_in)`), zero biases.
- **Model as written:** the `predict` function collapses the 11 features with `sum_without_collapse` (an *unweighted* sum across features), then applies `W1·x + b → ReLU → mean_without_collapse → W2·x + b → ReLU → W3·x + b → sigmoid × 10` (a sigmoid scaled to the 0–10 quality range). Because of the sum/mean reductions, the weight matrices are all 1×1, so this is effectively a scalar-in, scalar-out network. It meets the assignment's requirements (a hidden layer and an activation) but does not learn per-feature weights.
- **Training:** `Adam(lr=1e-3)` for 1,000 epochs on MSE loss, using `BearTensor` all the way through, so `loss.backward()` drives the custom engine.
- **Results (from the notebook output):**
  - Training MSE falls from about **1.06 → 0.653** and plateaus by about epoch 600.
  - **Test MSE = 0.650**, essentially equal to the training MSE, so there is no sign of overfitting. The assignment threshold was MSE ≤ 2.0.
- A reasonable follow-up would be a full `11 → h → 1` network so the model can weight individual features. As built, the low capacity explains why the loss plateaus near 0.65.
