import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GptModel } from './gpt.ts';
import { MlpModel } from './mlp.ts';
import { BigramModel } from './bigram.ts';
import { configFor } from './factory.ts';
import type { ModelConfig } from './base.ts';
import { Rng } from '../core/rng.ts';

/**
 * The tests that matter most in this project.
 *
 * A hand-derived backward pass has exactly one dangerous failure mode: it runs,
 * it trains, and it is subtly wrong — the loss curve just looks worse than it
 * should, which is very easy to blame on hyperparameters. So every parameter
 * gradient of every architecture is checked here against a numerical
 * derivative of the architecture's own forward pass.
 */

interface Setup {
  model: GptModel | MlpModel | BigramModel;
  tokens: Int32Array;
  targets: Int32Array;
  B: number;
  T: number;
  vocabSize: number;
}

function setup(kind: 'gpt' | 'mlp' | 'bigram', cfg: Partial<ModelConfig>, seed = 5): Setup {
  const vocabSize = 9;
  const config = configFor(kind, vocabSize, { seed, ...cfg });
  const model = kind === 'gpt' ? new GptModel(config) : kind === 'mlp' ? new MlpModel(config) : new BigramModel(config);
  const rng = new Rng(seed + 100);
  const B = 2;
  const T = config.blockSize;
  const tokens = Int32Array.from({ length: B * T }, () => rng.int(vocabSize));
  const targets = Int32Array.from({ length: B * T }, () => rng.int(vocabSize));
  return { model, tokens, targets, B, T, vocabSize };
}

/**
 * Compare every named parameter's analytic gradient against central finite
 * differences of the loss, on a few randomly chosen entries.
 */
function assertGradientsMatch(setupIn: Setup, names: string[], samplesPerParam = 3, eps = 2e-2): void {
  const { model, tokens, targets, B, T } = setupIn;
  model.zeroGrad();
  const baseLoss = model.forward(tokens, targets, B, T);
  model.backward();
  const picker = new Rng(4242);
  let checked = 0;

  for (const name of names) {
    const p = model.params.find((q) => q.name === name);
    assert.ok(p, `no such parameter: ${name}`);
    const analytic = p!.grad.slice();
    for (let s = 0; s < samplesPerParam; s++) {
      const idx = picker.int(p!.data.length);
      const orig = p!.data[idx];
      p!.data[idx] = orig + eps;
      const plus = model.forward(tokens, targets, B, T);
      p!.data[idx] = orig - eps;
      const minus = model.forward(tokens, targets, B, T);
      p!.data[idx] = orig;
      const numeric = (plus - minus) / (2 * eps);
      const a = analytic[idx];
      const tol = 2e-3 + 0.05 * Math.abs(numeric);
      assert.ok(
        Math.abs(a - numeric) <= tol,
        `${name}[${idx}]: analytic=${a.toExponential(4)} numeric=${numeric.toExponential(4)} (base loss ${baseLoss.toFixed(4)})`,
      );
      checked++;
    }
  }
  assert.ok(checked >= names.length, 'expected to check at least one entry per parameter');
}

test('gpt: every parameter gradient matches finite differences (1 layer, 2 heads)', () => {
  const s = setup('gpt', { blockSize: 5, nEmbd: 8, nLayer: 1, nHead: 2 });
  assertGradientsMatch(s, [
    'wte',
    'wpe',
    'h0.ln1.weight',
    'h0.ln1.bias',
    'h0.attn.qkv.weight',
    'h0.attn.qkv.bias',
    'h0.attn.proj.weight',
    'h0.ln2.weight',
    'h0.mlp.fc.weight',
    'h0.mlp.fc.bias',
    'h0.mlp.proj.weight',
    'lnf.weight',
    'lnf.bias',
  ]);
});

test('gpt: gradients still match with two layers and four heads', () => {
  const s = setup('gpt', { blockSize: 4, nEmbd: 8, nLayer: 2, nHead: 4 });
  assertGradientsMatch(s, ['wte', 'wpe', 'h0.attn.qkv.weight', 'h1.attn.qkv.weight', 'h1.mlp.fc.weight', 'h1.mlp.proj.weight', 'lnf.weight'], 2);
});

test('mlp: every parameter gradient matches finite differences', () => {
  const s = setup('mlp', { blockSize: 6, nEmbd: 8, nLayer: 1, nHead: 1 });
  assertGradientsMatch(s, ['wte', 'mlp.fc1.weight', 'mlp.fc1.bias', 'mlp.fc2.weight', 'mlp.fc2.bias']);
});

test('bigram: every parameter gradient matches finite differences', () => {
  const s = setup('bigram', { blockSize: 4, nEmbd: 8, nLayer: 1, nHead: 1 });
  assertGradientsMatch(s, ['bigram.table'], 5);
});

test('initial loss is close to ln(vocab) for every architecture', () => {
  const expected = Math.log(9);
  for (const kind of ['bigram', 'mlp', 'gpt'] as const) {
    const s = setup(kind, { blockSize: 4, nEmbd: 8, nLayer: 1, nHead: 2 }, 3);
    const loss = s.model.forward(s.tokens, s.targets, s.B, s.T);
    assert.ok(
      Math.abs(loss - expected) < 0.35,
      `${kind}: initial loss ${loss.toFixed(3)} should be within 0.35 of ln(9)=${expected.toFixed(3)}`,
    );
  }
});

test('attention is causal: no weight above the diagonal, every row sums to 1', () => {
  const s = setup('gpt', { blockSize: 6, nEmbd: 8, nLayer: 2, nHead: 2 });
  s.model.forward(s.tokens, s.targets, s.B, s.T);
  assert.equal(s.model.attention.length, 4, 'two layers x two heads of attention maps');
  for (const map of s.model.attention) {
    const t = map.t;
    for (let i = 0; i < t; i++) {
      let sum = 0;
      for (let j = 0; j < t; j++) {
        const w = map.weights[i * t + j];
        if (j > i) assert.equal(w, 0, `layer ${map.layer} head ${map.head}: position ${i} attends to future position ${j}`);
        else {
          assert.ok(w >= 0, 'attention weights are non-negative');
          sum += w;
        }
      }
      assert.ok(Math.abs(sum - 1) < 1e-4, `row ${i} of the attention matrix must sum to 1 (got ${sum})`);
    }
  }
});

test('the same seed and the same batch give bit-identical losses', () => {
  const a = setup('gpt', { blockSize: 5, nEmbd: 8, nLayer: 1, nHead: 2 }, 7);
  const b = setup('gpt', { blockSize: 5, nEmbd: 8, nLayer: 1, nHead: 2 }, 7);
  const la = a.model.forward(a.tokens, a.targets, a.B, a.T);
  const lb = b.model.forward(b.tokens, b.targets, b.B, b.T);
  assert.equal(la, lb);
});

test('an untrained model produces the same sampled ids twice', async () => {
  const { generate } = await import('./sample.ts');
  const s = setup('gpt', { blockSize: 8, nEmbd: 8, nLayer: 1, nHead: 2 }, 11);
  const one = generate(s.model, [1, 2, 3], { maxNew: 12, temperature: 1, topK: 5, seed: 99 });
  const two = generate(s.model, [1, 2, 3], { maxNew: 12, temperature: 1, topK: 5, seed: 99 });
  assert.deepEqual(one, two);
  assert.equal(one.length, 12);
  for (const id of one) assert.ok(id >= 0 && id < 9, `sampled id ${id} outside the vocabulary`);
});
