# Architecture

A map for reading `src/`, in the order data actually flows. Everything here is
short on purpose: the point of the project is that a person can hold the whole
thing in their head.

```
corpus text
   │  src/data/tokenizer.ts        characters → ids (vocab = distinct chars)
   ▼
 Int32Array batch  ──────────────────────────────────────────────┐
   │  src/model/factory.ts        config for bigram | mlp | gpt  │
   ▼                                                             │
 model.forward(tokens, targets, B, T)      ← src/model/{gpt,mlp,bigram}.ts
   │      stores: logits, probs, per-layer activations, attention maps
   ▼
 cross-entropy → lastLoss
   │  src/train/optimizer.ts      clipping, Adam, warmup + cosine schedule
   ▼
 model.backward()                          ← hand-written, no tape
   │      fills every Param.grad
   ▼
 Adam step → params updated → repeat
```

## Tensors, and the one convention that matters

`src/core/mat.ts` is a `Mat`: `{ rows, cols, data: Float32Array }`, row-major,
no strides, no views, no broadcasting. Three kernels cover the whole project:

| kernel | computes | used by |
| --- | --- | --- |
| `matmulNN(a, b)` | `a @ b` | attention scores (`q @ kᵀ` is done as `NT`) |
| `matmulNT(a, b)` | `a @ bᵀ` | **every** linear layer — weights are stored `[out, in]` |
| `matmulTN(a, b)` | `aᵀ @ b` | backward passes for weight gradients |

Storing weights as `[out, in]` (the PyTorch `Linear` convention) means the forward
pass is always `x @ Wᵀ`: `matmulNT`. Getting this backwards is the single easiest
mistake in the codebase — it happened during development, and
`src/model/gpt.test.ts` is what caught it, because a transposed matmul produces
gradients that pass a loss sanity check but fail finite differences.

Every kernel takes an optional `out` and an `accumulate` flag, so backward passes
can add into an existing gradient buffer instead of allocating. The training loop
allocates nothing at all after the first step.

## forward / backward, by hand

`src/model/gpt.ts` is a decoder-only transformer, pre-norm, with tied embeddings:

```
x  = wte[token] + wpe[position]
for each block:
    x += attn(ln1(x))            multi-head causal attention, softmax over keys ≤ t
    x += mlp(ln2(x))             linear → gelu → linear
lnf = ln2-style final norm(x)
logits = lnf @ wteᵀ              (tied: the same matrix as the embedding)
```

`forward()` writes into preallocated buffers and, when a batch is present, records
the attention weights of the **first sequence** for the visualiser. `backward()`
walks the same layers in reverse, with the softmax/cross-entropy gradient folded
into a single `dlogits` scatter, and computes each parameter's gradient into its
own `Param.grad`.

Attention and MLP backward passes avoid temporaries by streaming over rows with
two scratch rows (`rowDatt`, `rowDs`) instead of materialising `T × T` matrices.

## Two engines, one oracle

`src/core/autograd.ts` is an independent, tape-based reverse-mode engine: `Var`
nodes, each operation closes over its parents and a local backward function; every
`forward()` push records a `TapeNode`, and `backward()` walks the tape in reverse.

It exists to *check* the hand-written gradients, not to train:

```ts
// src/core/gradCheck.ts — central differences, float32-friendly step size
numericalGrad(f)   // (f(x+h) - f(x-h)) / 2h
compareGrad(...)   // relative error, with an absolute floor for near-zero grads
```

`src/model/gpt.test.ts` runs both against the same weights and asserts they agree
with each other and with numerical differentiation.

## Shapes, caching, and why generation is fast enough

`GptModel.ensure(N, B, T)` installs the activation buffers for a shape. Training
uses `(B*T, batchSize, blockSize)`; generation uses `(blockSize, 1, blockSize)`.
A small map keeps the last two shapes alive, so switching between training and
sampling costs nothing after the first switch — the alternative (reallocating
every activation set per generated character) dominated the runtime.

`Generator` (in `src/model/sample.ts`) keeps the window at exactly `blockSize`,
padding on the right. Causal attention means the row that predicts the next
character cannot see that padding, so the trick is free, and one fixed shape is
all generation ever needs.

## The browser side

- `src/ui/worker.ts` owns the model and runs a cooperative loop: up to 45ms of
  training steps per tick, then a couple of characters of generation, then a
  message to the page. The tab stays at 60fps because none of this is on the main
  thread.
- Metrics go every tick; attention, PCA embedding and gradient norms are
  throttled to 320ms / 2.2s / 900ms because they are only worth redrawing at
  human speed.
- `src/ui/charts.ts` draws everything with `CanvasRenderingContext2D`: the loss
  curve on a log axis with the uniform-guessing baseline as a reference line, the
  attention heatmap, the PCA scatter of the embedding matrix, the gradient-norm
  bars.
- `src/model/graph.ts` and `src/model/flops.ts` derive the "forward graph" panel
  and the FLOP counters from the model config, so the numbers cannot drift away
  from the code.

## Deliberate omissions

- No KV cache (one inference path, identical to training).
- No BPE (a character vocabulary keeps the tokenizer readable at 63 lines).
- No WebGPU/WASM (the point is that plain JavaScript on a CPU is enough to show
  the mechanism).
