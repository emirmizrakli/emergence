# emergence

**A transformer, a reverse-mode autodiff engine and a training loop — written from scratch, no runtime dependencies — that train live in your browser tab.**

[**▶ watch it train**](https://emirmizrakli.github.io/emergence/) · [run the command line version](#run-it) · [read the code map](#what-is-actually-in-here)

Open the page and a character-level language model starts learning English in front of you. Not a recording, not a pre-rendered chart: the loss curve, the attention heads, the embedding space, the gradient norms and the text it writes are all recomputed from the weights as they change.

```
$ node src/cli/train.ts --model gpt --steps 40 --batch 32 --block 48 --embd 64 --layers 4 --heads 4
step    40 | loss 2.9135 | val 2.8843 | lr 3.05e-4 | 1,133 tok/s
  sample @40: COOO ⏎ G ⏎ Xsod euon pe iesice, ands tae y g toy hy knea' here m weuinor f mhliteo'iey
              thvyr s t ths tine  ires   st tthe he ⏎ Tt oke aae ed thh ru
```

Forty steps, and the shapes of English words are already there — out of a model whose weights started as random noise.

---

## Why this exists

Every ML tool in a portfolio is a `pip install` away from looking like a wrapper. `nn.Transformer` is four lines and proves nothing about whether the person writing it understands attention, backpropagation, or why a training run diverges.

So this project deliberately removes every abstraction that would hide the answer. There is no PyTorch, no TensorFlow, no numpy, no ML library of any kind — not even a charting library. What is left is:

- a `Float32Array` matrix library with the three matmul layouts that matter (`NN`, `NT`, `TN`),
- a tape-based reverse-mode autodiff engine that can differentiate anything you build out of its primitives,
- a hand-written decoder-only transformer with explicit forward *and* backward passes,
- Adam with warmup, cosine decay and gradient clipping,
- a trainer, a checkpoint format, a CLI, and a worker-thread UI.

Both engines exist on purpose. The fast path is hand-written kernels, because tensor gymnastics through a generic tape is 10× slower. But hand-written gradients can be quietly wrong — so an independent autodiff implementation doubles as an oracle, and both are checked against numerical differentiation (`src/core/gradCheck.ts`). If a hand-written backward pass disagrees with finite differences, the test suite fails. That is the whole safety argument, and it is enforced by tests instead of asserted in a README.

## What is actually in here

| area | files | what it does |
| --- | --- | --- |
| `src/core/` | `mat.ts`, `autograd.ts`, `rng.ts`, `gradCheck.ts` | matrix kernels, tape autodiff, seeded RNG (mulberry32 + Box–Muller), central-difference gradient checking |
| `src/model/` | `gpt.ts`, `mlp.ts`, `bigram.ts`, `factory.ts`, `param.ts`, `sample.ts`, `flops.ts`, `graph.ts` | the three-rung model ladder, a flat parameter/gradient view, sampling, FLOP accounting, forward-graph description |
| `src/train/` | `trainer.ts`, `optimizer.ts`, `pca.ts`, `viz.ts`, `checkpoint.ts` | training loop, Adam, power-iteration PCA, internal-state snapshots, JSON checkpoints |
| `src/ui/` | `worker.ts`, `main.ts`, `charts.ts`, `style.css` | the worker that owns the model, the page orchestration, and a canvas renderer with no dependencies |
| `src/cli/` | `train.ts` | headless training, with `--compare` to run all three architectures under an identical budget |

**3,945 lines** of source, **693 lines** of tests, **0** runtime dependencies. Node 23+ runs the TypeScript directly — the CLI has no build step, because Node strips types natively.

## Measured, not promised

`node src/cli/train.ts --compare --steps 600` trains all three rungs on the same corpus with the same budget. Real output, on one laptop CPU core:

| model | params | first loss | final loss | val loss | val perplexity | minutes |
| --- | --- | --- | --- | --- | --- | --- |
| bigram | 4,225 | 4.179 | 3.183 | 3.205 | 24.7 | 0.0 |
| mlp | 151,936 | 4.174 | 2.410 | 2.516 | 12.4 | 6.2 |
| gpt | 106,304 | 4.170 | **2.287** | **2.393** | **11.0** | 4.9 |

The transformer is 30% *smaller* than the MLP and still wins: 106,304 parameters against 151,936, val perplexity 11.0 against 12.4. A fixed window treats every character in it as equally important; a causal mask does not. Full setup, the honest reading of each column, and a divergence that was measured and discarded are in [`docs/benchmarks.md`](docs/benchmarks.md).

## The three things worth reading

### 1. Two engines, and a test that makes them agree

`src/core/autograd.ts` builds a tape: each operation records its parents and a local backward closure, and `backward()` walks it in reverse. `src/core/gpt.ts` does the same math by hand, with no allocation in the inner loops.

The hand-written version is what trains. The tape is the oracle. `src/model/gpt.test.ts` takes a real batch, runs the exact forward pass the trainer uses, then compares **every** parameter's gradient against finite differences — plus a full comparison against the tape-based engine. A subtly wrong transpose in the output projection (the mistake this project actually made) cannot survive that.

### 2. A model ladder, because the failure is the point

The same trainer and the same corpus run three architectures:

| | what it can express | how it fails |
| --- | --- | --- |
| **bigram** | one character of context | plateaus at ~2.4 loss: it can learn letter frequencies and nothing else |
| **fixed-window MLP** | the last 8 characters, order-blinded by a sum | beats the bigram, then stalls — it has no idea *where* a character was |
| **transformer** | every position attends to every earlier position | keeps falling, and the attention panel shows what it learned to look at |

Switching between them in the UI, same corpus and same step budget, is the clearest possible demonstration of what attention actually buys you.

### 3. Generation that reuses the training buffers

Sampling runs with no KV cache: each new character re-runs the whole forward pass, so the inference path is byte-for-byte the training path and there is no second implementation to drift. To keep that cheap, the generation window is *always* exactly `blockSize` long — live tokens left, padding right. Causality makes the padding free (row `n-1` cannot see the tokens after it), and holding one fixed shape means generation reuses the same activation buffers as training instead of allocating a new set per character.

The worker also trains and samples on a cooperative schedule inside a 45ms-per-tick budget, so the page never blocks, and text streams in a couple of characters at a time while the loss keeps dropping.

## Verified, not asserted

```bash
npm test        # 35 tests, node --test — no test framework, no config
```

| test file | what it pins down |
| --- | --- |
| `src/core/mat.test.ts` | every kernel against a naive triple-loop reference, including the `NT`/`TN` transposed layouts and accumulation |
| `src/core/autograd.test.ts` | each tape operation against its own forward function's finite differences |
| `src/model/gpt.test.ts` | all three architectures: every parameter gradient vs. central differences, and vs. the autodiff oracle |
| `src/train/trainer.test.ts` | tokenizer round-trip, the warmup/cosine learning-rate schedule, and checkpoint save → restore → identical logits |

The gradient checks use central differences with a step size tuned for float32, and a relative-error tolerance that a wrong transpose blows straight through.

## Numbers, from this machine

The CLI reports tokens/second, parameter counts, held-out loss and perplexity for the exact same corpus and step budget:

```bash
node src/cli/train.ts --compare --steps 600
```

The demo page's `medium` preset runs a 677k-parameter model; `small` (the bundled checkpoint's shape) is 207k parameters, 4 blocks, 4 heads, 48-character context, trained to a held-out loss around 1.9 on tiny shakespeare.

Nothing here needs a GPU, and nothing here uses one: it is single-threaded JavaScript on your CPU, which is exactly why the numbers on the page are believable.

## Run it

```bash
git clone https://github.com/emirmizrakli/emergence
cd emergence
npm install

npm run dev              # the live page on localhost
npm test                 # the whole suite
npm run typecheck        # strict TypeScript, no emit

# headless training, same code as the page
node src/cli/train.ts --model gpt --steps 4000 --out checkpoints/gpt.json
node src/cli/train.ts --compare --steps 600      # all three architectures, one table

npm run sync             # fetch the corpus and stage the demo assets
npm run build            # static build into dist/
```

Try it with your own text: `npm run dev`, set *corpus* to **your own text**, paste something, and the tokenizer is rebuilt on it — the model then learns your text from scratch, from random noise, in front of you.

## Deploying the demo page

```bash
npm run deploy     # builds, then force-pushes dist/ to a gh-pages branch
```

One-time setup: **Settings → Pages → Source: Deploy from a branch → `gh-pages` → `/ (root)`.**
The page is static, so it needs no serverless functions, no build minutes and no
runner: publishing is a `git push`. (The Actions workflow in `.github/workflows/`
is a manual typecheck + test runner, intentionally not a deployer — nothing here
depends on CI to be released.)

## What this is not

Being clear about the limits is part of the point:

- **CPU only.** No GPU, no WebGPU, no WASM SIMD. Batches are tiny and models are small. This is a laboratory, not a training service.
- **Character-level.** No BPE. The vocabulary is the distinct characters of the corpus (65 for Shakespeare).
- **No KV cache.** Generation is deliberately O(n²) per character in order to keep one inference path.
- **Small models.** 30k–700k parameters. It learns English prose, not facts.

## License

MIT © Emir Mızraklı
