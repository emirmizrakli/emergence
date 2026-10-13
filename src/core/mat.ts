/**
 * Dense 2-D matrices over Float32Array, plus the numeric kernels the whole
 * project is built on.
 *
 * Design notes
 * ------------
 * - One flat Float32Array per matrix, row-major. No views, no strides: index
 *   math stays obvious and the kernels stay fast.
 * - `matmul*` come in NN / NT / TN flavours instead of relying on an explicit
 *   transpose, because transposing allocates and copies on every step.
 * - Kernels take an optional `out` and an `accumulate` flag. Training reuses
 *   preallocated buffers, so a step allocates almost nothing.
 * - Every kernel here has a matching gradient test in src/core/mat.test.ts
 *   (finite differences), which is how the maths is kept honest.
 */

export interface Mat {
  rows: number;
  cols: number;
  data: Float32Array;
}

export function mat(rows: number, cols: number, data?: Float32Array): Mat {
  return { rows, cols, data: data ?? new Float32Array(rows * cols) };
}

export function zeros(rows: number, cols: number): Mat {
  return mat(rows, cols);
}

/** xavier-uniform-ish init for a [rows, cols] matrix. */
export function filled(rows: number, cols: number, fn: (i: number, j: number) => number): Mat {
  const data = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) data[i * cols + j] = fn(i, j);
  }
  return mat(rows, cols, data);
}

export function cloneMat(a: Mat): Mat {
  return mat(a.rows, a.cols, a.data.slice());
}

/** out = 0 */
export function clear(out: Mat): void {
  out.data.fill(0);
}

/** out = a */
export function copyInto(out: Mat, a: Mat): void {
  out.data.set(a.data);
}

/** out += a (elementwise, same shape) */
export function addInto(out: Mat, a: Mat): void {
  const o = out.data;
  const x = a.data;
  for (let i = 0; i < o.length; i++) o[i] += x[i];
}

/** out += a * s */
export function addScaledInto(out: Mat, a: Mat, s: number): void {
  const o = out.data;
  const x = a.data;
  for (let i = 0; i < o.length; i++) o[i] += x[i] * s;
}

/** out *= s */
export function scale(out: Mat, s: number): void {
  const o = out.data;
  for (let i = 0; i < o.length; i++) o[i] *= s;
}

/**
 * C = A @ B for A [N,K], B [K,M].
 * `accumulate` adds into `out` instead of overwriting it.
 */
export function matmulNN(a: Mat, b: Mat, out?: Mat, accumulate = false): Mat {
  const N = a.rows;
  const K = a.cols;
  if (b.rows !== K) throw new Error(`matmulNN shape mismatch: [${a.rows},${a.cols}] x [${b.rows},${b.cols}]`);
  const M = b.cols;
  const c = out ?? zeros(N, M);
  if (c.rows !== N || c.cols !== M) throw new Error(`matmulNN out shape mismatch: expected [${N},${M}]`);
  const A = a.data;
  const B = b.data;
  const C = c.data;
  if (!accumulate) C.fill(0);
  for (let i = 0; i < N; i++) {
    const ai = i * K;
    const ci = i * M;
    for (let k = 0; k < K; k++) {
      const aik = A[ai + k];
      if (aik === 0) continue;
      const bk = k * M;
      for (let j = 0; j < M; j++) C[ci + j] += aik * B[bk + j];
    }
  }
  return c;
}

/** C = A @ B^T for A [N,K], B [M,K] -> C [N,M]. Used by Linear layers (weight is [out,in]). */
export function matmulNT(a: Mat, b: Mat, out?: Mat, accumulate = false): Mat {
  const N = a.rows;
  const K = a.cols;
  if (b.cols !== K) throw new Error(`matmulNT shape mismatch: [${a.rows},${a.cols}] x [${b.rows},${b.cols}]^T`);
  const M = b.rows;
  const c = out ?? zeros(N, M);
  if (c.rows !== N || c.cols !== M) throw new Error(`matmulNT out shape mismatch: expected [${N},${M}]`);
  const A = a.data;
  const B = b.data;
  const C = c.data;
  if (!accumulate) C.fill(0);
  for (let i = 0; i < N; i++) {
    const ai = i * K;
    const ci = i * M;
    for (let j = 0; j < M; j++) {
      const bj = j * K;
      let s = 0;
      for (let k = 0; k < K; k++) s += A[ai + k] * B[bj + k];
      C[ci + j] += s;
    }
  }
  return c;
}

/** C = A^T @ B for A [K,N], B [K,M] -> C [N,M]. Used by weight gradients. */
export function matmulTN(a: Mat, b: Mat, out?: Mat, accumulate = false): Mat {
  const K = a.rows;
  if (b.rows !== K) throw new Error(`matmulTN shape mismatch: [${a.rows},${a.cols}]^T x [${b.rows},${b.cols}]`);
  const N = a.cols;
  const M = b.cols;
  const c = out ?? zeros(N, M);
  if (c.rows !== N || c.cols !== M) throw new Error(`matmulTN out shape mismatch: expected [${N},${M}]`);
  const A = a.data;
  const B = b.data;
  const C = c.data;
  if (!accumulate) C.fill(0);
  for (let k = 0; k < K; k++) {
    const ak = k * N;
    const bk = k * M;
    for (let i = 0; i < N; i++) {
      const aki = A[ak + i];
      if (aki === 0) continue;
      const ci = i * M;
      for (let j = 0; j < M; j++) C[ci + j] += aki * B[bk + j];
    }
  }
  return c;
}

/** out[n, :] += bias for every row. */
export function addBias(y: Mat, bias: Float32Array): void {
  const { rows, cols, data } = y;
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    for (let j = 0; j < cols; j++) data[off + j] += bias[j];
  }
}

/** db[j] += sum over rows of dy[:, j] */
export function sumRowsInto(db: Float32Array, dy: Mat): void {
  const { rows, cols, data } = dy;
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    for (let j = 0; j < cols; j++) db[j] += data[off + j];
  }
}

/** Row-wise softmax in place, over the full width of each row. */
export function softmaxRowsInPlace(m: Mat): void {
  const { rows, cols, data } = m;
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    let max = -Infinity;
    for (let j = 0; j < cols; j++) if (data[off + j] > max) max = data[off + j];
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      const v = Math.exp(data[off + j] - max);
      data[off + j] = v;
      sum += v;
    }
    const inv = sum > 0 ? 1 / sum : 0;
    for (let j = 0; j < cols; j++) data[off + j] *= inv;
  }
}

/** GELU (tanh approximation, as used by GPT-2). */
export function geluInPlace(m: Mat): void {
  const d = m.data;
  const c = 0.7978845608028654; // sqrt(2/pi)
  for (let i = 0; i < d.length; i++) {
    const x = d[i];
    const inner = c * (x + 0.044715 * x * x * x);
    const t = Math.tanh(inner);
    d[i] = 0.5 * x * (1 + t);
  }
}

/** dy = dx * gelu'(x), given the *input* x and incoming gradient dy. */
export function geluBackwardInPlace(dy: Mat, x: Float32Array): void {
  const d = dy.data;
  const c = 0.7978845608028654;
  for (let i = 0; i < d.length; i++) {
    const v = x[i];
    const v3 = v * v * v;
    const inner = c * (v + 0.044715 * v3);
    const t = Math.tanh(inner);
    const sech2 = 1 - t * t;
    const dInner = c * (1 + 3 * 0.044715 * v * v);
    d[i] *= 0.5 * (1 + t) + 0.5 * v * sech2 * dInner;
  }
}

export interface LayerNormCache {
  mean: Float32Array;
  rstd: Float32Array;
}

export function makeLayerNormCache(rows: number, cols: number): LayerNormCache {
  void cols;
  return { mean: new Float32Array(rows), rstd: new Float32Array(rows) };
}

/** y = (x - mean) / sqrt(var + eps) * g + b, per row. */
export function layernormForward(x: Mat, g: Float32Array, b: Float32Array, out: Mat, cache: LayerNormCache, eps = 1e-5): void {
  const { rows, cols, data } = x;
  const o = out.data;
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    let mean = 0;
    for (let j = 0; j < cols; j++) mean += data[off + j];
    mean /= cols;
    let variance = 0;
    for (let j = 0; j < cols; j++) {
      const d = data[off + j] - mean;
      variance += d * d;
    }
    variance /= cols;
    const rstd = 1 / Math.sqrt(variance + eps);
    cache.mean[i] = mean;
    cache.rstd[i] = rstd;
    for (let j = 0; j < cols; j++) o[off + j] = (data[off + j] - mean) * rstd * g[j] + b[j];
  }
}

/**
 * Given dy (gradient wrt output), accumulate dx, dg, db.
 * dx = rstd * (dgHat - mean(dgHat) - xHat * mean(dgHat * xHat))
 */
export function layernormBackward(
  dy: Mat,
  x: Mat,
  g: Float32Array,
  cache: LayerNormCache,
  dx: Mat,
  dg: Float32Array,
  db: Float32Array,
): void {
  const { rows, cols, data } = x;
  const dyd = dy.data;
  const dxd = dx.data;
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    const mean = cache.mean[i];
    const rstd = cache.rstd[i];
    let sum1 = 0;
    let sum2 = 0;
    for (let j = 0; j < cols; j++) {
      const xHat = (data[off + j] - mean) * rstd;
      const dgHat = dyd[off + j] * g[j];
      sum1 += dgHat;
      sum2 += dgHat * xHat;
    }
    sum1 /= cols;
    sum2 /= cols;
    for (let j = 0; j < cols; j++) {
      const xHat = (data[off + j] - mean) * rstd;
      const dgHat = dyd[off + j] * g[j];
      dxd[off + j] += rstd * (dgHat - sum1 - xHat * sum2);
      dg[j] += dyd[off + j] * xHat;
      db[j] += dyd[off + j];
    }
  }
}

/**
 * Mean cross-entropy over N rows of logits [N, V] against `targets`.
 * Optionally writes the row-wise softmax probabilities into `probs`.
 */
export function crossEntropy(logits: Mat, targets: Int32Array, probs?: Mat): number {
  const { rows, cols, data } = logits;
  let total = 0;
  const p = probs?.data;
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    let max = -Infinity;
    for (let j = 0; j < cols; j++) if (data[off + j] > max) max = data[off + j];
    let sum = 0;
    for (let j = 0; j < cols; j++) {
      const v = Math.exp(data[off + j] - max);
      if (p) p[off + j] = v;
      sum += v;
    }
    const logSum = Math.log(sum) + max;
    total += logSum - data[off + targets[i]];
    if (p) {
      const inv = 1 / sum;
      for (let j = 0; j < cols; j++) p![off + j] *= inv;
    }
  }
  return total / rows;
}

/** dlogits = (probs - onehot(targets)) / N */
export function crossEntropyBackward(dlogits: Mat, probs: Mat, targets: Int32Array): void {
  const { rows, cols } = dlogits;
  const d = dlogits.data;
  const p = probs.data;
  const scale = 1 / rows;
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    for (let j = 0; j < cols; j++) d[off + j] = p[off + j] * scale;
    d[off + targets[i]] -= scale;
  }
}
