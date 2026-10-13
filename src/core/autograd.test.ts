import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Var } from './autograd.ts';
import {
  crossEntropy,
  geluInPlace,
  layernormBackward,
  layernormForward,
  makeLayerNormCache,
  mat as makeMat,
  matmulNT,
  addBias,
  zeros,
} from './mat.ts';
import { Rng } from './rng.ts';
import { compareGrad, describeGrad, numericalGrad } from './gradCheck.ts';

/**
 * Every op in the tape engine is checked against finite differences of its own
 * forward pass. That is what lets the same engine serve as the reference oracle
 * for the hand-written kernels in mat.ts.
 */

test('Var.add gradient', () => {
  const rng = new Rng(11);
  const x0 = Float32Array.from({ length: 6 }, () => rng.normal(1));
  const y0 = Float32Array.from({ length: 6 }, () => rng.normal(1));
  const build = (x: Float32Array, y: Float32Array): Var =>
    new Var(x.slice(), 2, 3).add(new Var(y.slice(), 2, 3)).meanSquare();
  const numeric = numericalGrad((v) => build(v.subarray(0, 6), v.subarray(6)).data[0], Float32Array.from([...x0, ...y0]));
  const x = new Var(x0.slice(), 2, 3);
  const y = new Var(y0.slice(), 2, 3);
  x.add(y).meanSquare().backward();
  const cmp = compareGrad([...x.grad, ...y.grad], Array.from(numeric));
  assert.ok(cmp.ok, describeGrad(cmp));
});

test('Var.matmul gradient', () => {
  const rng = new Rng(12);
  const a0 = Float32Array.from({ length: 12 }, () => rng.normal(0.7));
  const b0 = Float32Array.from({ length: 12 }, () => rng.normal(0.7));
  const build = (a: Float32Array, b: Float32Array): Var =>
    new Var(a.slice(), 4, 3).matmul(new Var(b.slice(), 3, 4)).meanSquare();
  const numeric = numericalGrad((v) => build(v.subarray(0, 12), v.subarray(12)).data[0], Float32Array.from([...a0, ...b0]));
  const a = new Var(a0.slice(), 4, 3);
  const b = new Var(b0.slice(), 3, 4);
  a.matmul(b).meanSquare().backward();
  const cmp = compareGrad([...a.grad, ...b.grad], Array.from(numeric));
  assert.ok(cmp.ok, describeGrad(cmp));
});

test('Var.softmaxRows gradient', () => {
  const rng = new Rng(13);
  const x0 = Float32Array.from({ length: 12 }, () => rng.normal(1));
  const build = (x: Float32Array): Var => new Var(x.slice(), 3, 4).softmaxRows().meanSquare();
  const numeric = numericalGrad((v) => build(v).data[0], x0);
  const x = new Var(x0.slice(), 3, 4);
  x.softmaxRows().meanSquare().backward();
  const cmp = compareGrad(x.grad, numeric);
  assert.ok(cmp.ok, describeGrad(cmp));
});

test('Var.layernorm gradient for x, scale and shift', () => {
  const rng = new Rng(14);
  const rows = 4;
  const cols = 5;
  const C = rows * cols;
  const x0 = Float32Array.from({ length: C }, () => rng.normal(1));
  const g0 = Float32Array.from({ length: cols }, () => 1 + rng.normal(0.1));
  const b0 = Float32Array.from({ length: cols }, () => rng.normal(0.1));
  const packed = Float32Array.from([...x0, ...g0, ...b0]);
  const build = (v: Float32Array): Var =>
    new Var(v.slice(0, C), rows, cols).layernorm(new Var(v.slice(C, C + cols), 1, cols), new Var(v.slice(C + cols), 1, cols)).meanSquare();
  const numeric = numericalGrad((v) => build(v).data[0], packed);
  const x = new Var(x0.slice(), rows, cols);
  const g = new Var(g0.slice(), 1, cols);
  const b = new Var(b0.slice(), 1, cols);
  x.layernorm(g, b).meanSquare().backward();
  const cmp = compareGrad([...x.grad, ...g.grad, ...b.grad], numeric);
  assert.ok(cmp.ok, describeGrad(cmp));
});

test('Var.gelu gradient', () => {
  const rng = new Rng(15);
  const x0 = Float32Array.from({ length: 10 }, () => rng.normal(1.2));
  const build = (x: Float32Array): Var => new Var(x.slice(), 5, 2).gelu().meanSquare();
  const numeric = numericalGrad((v) => build(v).data[0], x0);
  const x = new Var(x0.slice(), 5, 2);
  x.gelu().meanSquare().backward();
  const cmp = compareGrad(x.grad, numeric);
  assert.ok(cmp.ok, describeGrad(cmp));
});

test('Var.addBias gradient', () => {
  const rng = new Rng(16);
  const rows = 3;
  const cols = 4;
  const y0 = Float32Array.from({ length: rows * cols }, () => rng.normal(1));
  const b0 = Float32Array.from({ length: cols }, () => rng.normal(0.4));
  const packed = Float32Array.from([...y0, ...b0]);
  const build = (v: Float32Array): Var => new Var(v.slice(0, rows * cols), rows, cols).addBias(new Var(v.slice(rows * cols), 1, cols)).meanSquare();
  const numeric = numericalGrad((v) => build(v).data[0], packed);
  const y = new Var(y0.slice(), rows, cols);
  const b = new Var(b0.slice(), 1, cols);
  y.addBias(b).meanSquare().backward();
  const cmp = compareGrad([...y.grad, ...b.grad], numeric);
  assert.ok(cmp.ok, describeGrad(cmp));
});

test('Var.T gradient', () => {
  const rng = new Rng(17);
  const x0 = Float32Array.from({ length: 12 }, () => rng.normal(1));
  const build = (x: Float32Array): Var => new Var(x.slice(), 3, 4).T.meanSquare();
  const numeric = numericalGrad((v) => build(v).data[0], x0);
  const x = new Var(x0.slice(), 3, 4);
  x.T.meanSquare().backward();
  const cmp = compareGrad(x.grad, numeric);
  assert.ok(cmp.ok, describeGrad(cmp));
});

test('both engines agree on a composite: layernorm(x @ Wᵀ + b)', () => {
  const rng = new Rng(18);
  const N = 6;
  const K = 5;
  const M = 4;
  const x0 = Float32Array.from({ length: N * K }, () => rng.normal(1));
  const w0 = Float32Array.from({ length: M * K }, () => rng.normal(0.6));
  const b0 = Float32Array.from({ length: M }, () => rng.normal(0.2));
  const g0 = Float32Array.from({ length: M }, () => 1 + rng.normal(0.05));
  const beta0 = Float32Array.from({ length: M }, () => rng.normal(0.05));

  // --- engine 1: hand-written kernels ---
  const x = makeMat(N, K, x0.slice());
  const W = makeMat(M, K, w0.slice());
  const bb = b0.slice();
  const g = g0.slice();
  const beta = beta0.slice();
  const pre = matmulNT(x, W);
  addBias(pre, bb);
  const ln = zeros(N, M);
  const cache = makeLayerNormCache(N, M);
  layernormForward(pre, g, beta, ln, cache, 1e-5);
  let lossValue = 0;
  for (let i = 0; i < ln.data.length; i++) lossValue += ln.data[i] * ln.data[i];
  lossValue /= ln.data.length;

  // backward by hand through the same composition
  const dLn = makeMat(N, M, Float32Array.from(ln.data, (v) => (2 * v) / ln.data.length));
  const dPre = zeros(N, M);
  const dg = new Float32Array(M);
  const dbeta = new Float32Array(M);
  layernormBackward(dLn, pre, g, cache, dPre, dg, dbeta);
  const dW = zeros(M, K);
  const dxMat = zeros(N, K);
  for (let n = 0; n < N; n++) {
    for (let m = 0; m < M; m++) {
      const gv = dPre.data[n * M + m];
      for (let k = 0; k < K; k++) {
        dW.data[m * K + k] += gv * x.data[n * K + k];
        dxMat.data[n * K + k] += gv * W.data[m * K + k];
      }
    }
  }
  const dbias = new Float32Array(M);
  for (let n = 0; n < N; n++) for (let m = 0; m < M; m++) dbias[m] += dPre.data[n * M + m];

  // --- engine 2: tape autodiff ---
  const vx = new Var(x0.slice(), N, K);
  const vW = new Var(w0.slice(), M, K);
  const vb = new Var(b0.slice(), 1, M);
  const vg = new Var(g0.slice(), 1, M);
  const vbeta = new Var(beta0.slice(), 1, M);
  const vLn = vx.matmul(vW.T).addBias(vb).layernorm(vg, vbeta);
  const vLoss = vLn.meanSquare();
  vLoss.backward();

  assert.ok(Math.abs(vLoss.data[0] - lossValue) < 1e-5, `forward mismatch: ${vLoss.data[0]} vs ${lossValue}`);

  const pairs: [string, ArrayLike<number>, ArrayLike<number>][] = [
    ['dW', dW.data, vW.grad],
    ['dx', dxMat.data, vx.grad],
    ['dbias', dbias, vb.grad],
    ['dg', dg, vg.grad],
    ['dbeta', dbeta, vbeta.grad],
  ];
  for (const [name, a, b] of pairs) {
    for (let i = 0; i < a.length; i++) {
      const scale = Math.max(1e-3, Math.abs(a[i]));
      assert.ok(Math.abs(a[i] - b[i]) < 1e-3 * scale + 1e-6, `${name}[${i}] ${a[i]} vs ${b[i]}`);
    }
  }
});

test('cross entropy of a uniform distribution equals ln(vocab)', () => {
  const V = 8;
  const rows = 3;
  const logits = makeMat(rows, V, new Float32Array(rows * V));
  const targets = Int32Array.from([0, 3, 7]);
  const loss = crossEntropy(logits, targets);
  assert.ok(Math.abs(loss - Math.log(V)) < 1e-5);
});

test('geluInPlace matches its own definition', () => {
  const m = makeMat(5, 1, Float32Array.from([-3, -0.5, 0, 0.5, 3]));
  const copy = Float32Array.from(m.data);
  geluInPlace(m);
  const c = 0.7978845608028654;
  for (let i = 0; i < 5; i++) {
    const x = copy[i];
    const want = 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
    assert.ok(Math.abs(m.data[i] - want) < 1e-6);
  }
});
