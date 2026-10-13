# Benchmarks

Three architectures, one corpus, one budget. Produced by the command in this
project — no numbers here were written by hand:

```bash
node src/cli/train.ts --compare --steps 600
```

## Setup

| | |
| --- | --- |
| corpus | `data/tiny-shakespeare.txt` — 1,115,394 characters, 65 distinct |
| split | 90% train (1,003,854 tokens) / 10% validation (111,540 tokens) |
| batch size | 32 windows per step |
| block size | 32 characters of context |
| optimizer | Adam, linear warmup 50 steps, cosine decay 3e-3 → 3e-4 |
| regulation | gradient-norm clip 1.0, weight decay 1e-2 |
| steps | 600 per model |
| machine | one laptop CPU core (Windows, Node 26) |

## Results

| model | params | first loss | final loss | val loss | val perplexity | minutes | tok/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bigram | 4,225 | 4.179 | 3.183 | 3.205 | 24.7 | 0.0 | 1,263,745 |
| mlp | 151,936 | 4.174 | 2.410 | 2.516 | 12.4 | 6.2 | 1,711 |
| gpt | 106,304 | 4.170 | 2.287 | 2.393 | 11.0 | 4.9 | 2,198 |

All three start at ≈ln(65) ≈ 4.17, which is what an untrained classifier over 65
characters should score. Everything after that is the architecture.

## How to read it

- **The bigram is the floor.** It sees one character and nothing else, so it can
  only learn letter frequencies. Perplexity 24.7 is the cost of knowing no
  context at all.
- **The MLP buys context the expensive way.** A fixed window of characters
  concatenated into one hidden layer: 151,936 parameters — 43% *more* than the
  transformer — for a worse result. A window treats every position the same and
  has no way to say "this character matters, that one doesn't".
- **The transformer gets more from less.** 106,304 parameters, 600 steps, val
  perplexity 11.0. The causal mask lets each position decide what to attend to,
  which is the whole point: it beats a model 1.4× its size at the same budget.
- **Wall-clock is not the comparison**, and `tok/s` is not either — each row
  measures its own batching, and the transformer does more arithmetic per token
  than the MLP. Params and val perplexity are the honest columns.

## A note on stability

The configuration shipped as the demo checkpoint (2 layers, block 32, lr 3e-3)
is stable: it is the `gpt` row above, and it keeps descending for thousands of
steps.

A deeper variant — 4 layers, block 48, the same lr — is not. On that run the
validation loss went 2.44 at step 1,750 → **3.57 at step 2,000** and never
recovered (3.43 at step 2,500, samples degraded to character soup). Grad-norm
clipping at 1.0 did not prevent it: Adam normalizes update magnitude, so the
clip cannot restrain a step that the moments already decided to take. That run
was discarded rather than shipped, and the parameters that *did* work are the
ones above. The lesson is in the repo instead of a footnote: at this size the
learning rate is doing more work than the architecture is.
