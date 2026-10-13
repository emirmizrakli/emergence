import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  mat,
  zeros,
  matmulNN,
  matmulNT,
  matmulTN,
  addBias,
  sumRowsInto,
  softmaxRowsInPlace,
  geluInPlace,
  geluBackwardInPlace,
  layernormForward,
  layernormBackward,
  makeLayerNormCache,
  crossEntropy,
  crossEntropyBackward,
  type Mat,
} from './mat.ts';
import { Rng } from './rng.ts';
import { compareGrad, describeGrad, numericalGrad } from './gradCheck.ts';

function randomMat(rows: number, cols: number, rng: Rng, scale = 1): Mat {
  return mat(rows, cols, Float32Array.from({ length: rows * cols }, () => rng.normal(scale)));
}

function naiveMatmul(a: Mat, b: Mat): Mat {
  const out = zeros(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < b.cols; j++) {
      let s = 0;
      for (let k = 0; k < a.cols; k++) s += a.data[i * a.cols + k] * b.data[k * b.cols + j];
      out.data[i * b.cols + j] = s;
    }
  }
  return out;
}

test('matmulNN matches a naive triple loop', () => {
  const rng = new Rng(1);
  const a = randomMat(5, 7, rng);
  const b = randomMat(7, 3, rng);
  const got = matmulNN(a, b);
  const want = naiveMatmul(a, b);
  for (let i = 0; i < want.data.length; i++) assert.ok(Math.abs(got.data[i] - want.data[i]) < 1e-5);
});

test('matmulNT is matmulNN against an explicit transpose', () => {
  const rng = new Rng(2);
  const a = randomMat(4, 6, rng);
  const b = randomMat(3, 6, rng);
  const explicit = zeros(6, 3);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 6; j++) explicit.data[j * 3 + i] = b.data[i * 6 + j];
  const got = matmulNT(a, b);
  const want = matmulNN(a, explicit);
  for (let i = 0; i < want.data.length; i++) assert.ok(Math.abs(got.data[i] - want.data[i]) < 1e-5);
});

test('matmulTN is matmulNN of the transposed left operand', () => {
  const rng = new Rng(3);
  const a = randomMat(5, 4, rng);
  const b = randomMat(5, 6, rng);
  const at = zeros(4, 5);
  for (let i = 0; i < 5; i++) for (let j = 0; j < 4; j++) at.data[j * 5 + i] = a.data[i * 4 + j];
  const got = matmulTN(a, b);
  const want = matmulNN(at, b);
  for (let i = 0; i < want.data.length; i++) assert.ok(Math.abs(got.data[i] - want.data[i]) < 1e-5);
});

test('matmul accumulate adds instead of overwriting', () => {
  const rng = new Rng(4);
  const a = randomMat(3, 3, rng);
  const b = randomMat(3, 3, rng);
  const out = zeros(3, 3);
  matmulNN(a, b, out);
  const once = out.data.slice();
  matmulNN(a, b, out, true);
  for (let i = 0; i < once.length; i++) assert.ok(Math.abs(out.data[i] - 2 * once[i]) < 1e-5);
});

test('matmul shape mismatches are loud', () => {
  const a = zeros(2, 3);
  const b = zeros(4, 2);
  assert.throws(() => matmulNN(a, b), /shape mismatch/);
});

test('gelu backward matches finite differences', () => {
  const rng = new Rng(5);
  const x = Float32Array.from({ length: 12 }, () => rng.normal(0.8));
  const dy0 = Float32Array.from({ length: 12 }, () => rng.normal(0.5));

  const loss = (v: Float32Array): number => {
    const m = mat(12, 1, v.slice());
    geluInPlace(m);
    let s = 0;
    for (let i = 0; i < 12; i++) s += m.data[i] * dy0[i];
    return s;
  };

  const numeric = numericalGrad(loss, x);
  const dy = mat(12, 1, dy0.slice());
  geluBackwardInPlace(dy, x);
  const cmp = compareGrad(Array.from(dy.data), Array.from(numeric));
  assert.ok(cmp.ok, describeGrad(cmp));
});

test('layernorm backward matches finite differences for dx, dg and db', () => {
  const rng = new Rng(6);
  const rows = 4;
  const cols = 6;
  const C = rows * cols;
  const x = Float32Array.from({ length: C }, () => rng.normal(1));
  const g = Float32Array.from({ length: cols }, () => 1 + rng.normal(0.1));
  const b = Float32Array.from({ length: cols }, () => rng.normal(0.1));
  const dy0 = Float32Array.from({ length: C }, () => rng.normal(0.7));

  const loss = (v: Float32Array): number => {
    const [xm, gm, bm] = split(v, C, cols);
    const out = zeros(rows, cols);
    layernormForward(xm, gm, bm, out, makeLayerNormCache(rows, cols));
    let s = 0;
    for (let i = 0; i < C; i++) s += out.data[i] * dy0[i];
    return s;
  };

  const packed = Float32Array.from([...x, ...g, ...b]);
  const numeric = numericalGrad(loss, packed);

  const xm = mat(rows, cols, x.slice());
  const gm = g.slice();
  const bm = b.slice();
  const dy = mat(rows, cols, dy0.slice());
  const dx = zeros(rows, cols);
  const dg = new Float32Array(cols);
  const db = new Float32Array(cols);
  const cache = makeLayerNormCache(rows, cols);
  const out = zeros(rows, cols);
  layernormForward(xm, gm, bm, out, cache);
  layernormBackward(dy, xm, gm, cache, dx, dg, db);

  const analytic = Float32Array.from([...dx.data, ...dg, ...db]);
  const cmp = compareGrad(analytic, numeric);
  assert.ok(cmp.ok, describeGrad(cmp));
});

test('addBias and sumRowsInto are consistent', () => {
  const rng = new Rng(7);
  const y = randomMat(3, 4, rng);
  const bias = Float32Array.from({ length: 4 }, () => rng.normal(0.3));
  addBias(y, bias);
  const db = new Float32Array(4);
  sumRowsInto(db, y);
  for (let j = 0; j < 4; j++) {
    let want = 0;
    for (let i = 0; i < 3; i++) want += y.data[i * 4 + j];
    assert.ok(Math.abs(db[j] - want) < 1e-5);
  }
});

test('softmaxRowsInPlace produces distributions', () => {
  const rng = new Rng(8);
  const m = randomMat(5, 9, rng, 2);
  softmaxRowsInPlace(m);
  for (let i = 0; i < 5; i++) {
    let s = 0;
    for (let j = 0; j < 9; j++) {
      assert.ok(m.data[i * 9 + j] >= 0);
      s += m.data[i * 9 + j];
    }
    assert.ok(Math.abs(s - 1) < 1e-5);
  }
});

test('cross entropy drops as the target logit grows, and its gradient is right', () => {
  const rng = new Rng(9);
  const rows = 4;
  const V = 6;
  const logits = randomMat(rows, V, rng);
  const targets = Int32Array.from({ length: rows }, () => rng.int(V));

  const loss = (v: Float32Array): number => crossEntropy(mat(rows, V, v.slice()), targets);
  const numeric = numericalGrad(loss, logits.data);

  const probs = zeros(rows, V);
  crossEntropy(logits, targets, probs);
  const dlogits = zeros(rows, V);
  crossEntropyBackward(dlogits, probs, targets);

  const cmp = compareGrad(Array.from(dlogits.data), Array.from(numeric));
  assert.ok(cmp.ok, describeGrad(cmp));

  // the gradient of a well-predicted target must be negative (loss goes down
  // when that logit goes up)
  const peaked = mat(2, 3, Float32Array.from([0, 0, 0, 5, 0, 0]));
  const peakedTargets = Int32Array.from([2, 0]);
  const p2 = zeros(2, 3);
  crossEntropy(peaked, peakedTargets, p2);
  const d2 = zeros(2, 3);
  crossEntropyBackward(d2, p2, peakedTargets);
  assert.ok(d2.data[2] < 0, 'up-weighting the correct target must lower the loss');
  assert.ok(d2.data[4] > 0, 'up-weighting a wrong target must raise the loss');
});

function split(packed: Float32Array, size: number, cols: number): [Mat, Float32Array, Float32Array] {
  const x = mat(size / cols, cols, packed.slice(0, size));
  const g = packed.slice(size, size + cols);
  const b = packed.slice(size + cols, size + 2 * cols);
  return [x, g, b];
}
