import type { Mat } from '../core/mat.ts';

/**
 * Two-dimensional PCA via power iteration, used to project the learned
 * embedding table down to something an eye can read. Deterministic (fixed
 * start vector) so the scatter plot does not jump around between frames.
 */
export function pca2(matrix: Mat, iterations = 60): Float32Array {
  const { rows, cols } = matrix;
  const out = new Float32Array(rows * 2);
  if (rows === 0 || cols === 0) return out;

  // centre
  const mean = new Float64Array(cols);
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    for (let j = 0; j < cols; j++) mean[j] += matrix.data[off + j];
  }
  for (let j = 0; j < cols; j++) mean[j] /= rows;

  const centred = new Float64Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    for (let j = 0; j < cols; j++) centred[off + j] = matrix.data[off + j] - mean[j];
  }

  // covariance (cols x cols)
  const cov = new Float64Array(cols * cols);
  for (let i = 0; i < rows; i++) {
    const off = i * cols;
    for (let a = 0; a < cols; a++) {
      const va = centred[off + a];
      if (va === 0) continue;
      for (let b = a; b < cols; b++) cov[a * cols + b] += va * centred[off + b];
    }
  }
  for (let a = 0; a < cols; a++) {
    for (let b = a; b < cols; b++) {
      const v = cov[a * cols + b];
      cov[a * cols + b] = v;
      cov[b * cols + a] = v;
    }
  }

  const component = (which: number): Float64Array => {
    let vec = new Float64Array(cols);
    // deterministic, non-degenerate start
    for (let i = 0; i < cols; i++) vec[i] = Math.sin(i * 12.9898 + which * 78.233);
    for (let iter = 0; iter < iterations; iter++) {
      const next = new Float64Array(cols);
      for (let a = 0; a < cols; a++) {
        let s = 0;
        const base = a * cols;
        for (let b = 0; b < cols; b++) s += cov[base + b] * vec[b];
        next[a] = s;
      }
      // deflate against previously found components
      for (let p = 0; p < which; p++) {
        const prev = prevComponents[p];
        let dot = 0;
        for (let i = 0; i < cols; i++) dot += next[i] * prev[i];
        for (let i = 0; i < cols; i++) next[i] -= dot * prev[i];
      }
      let norm = 0;
      for (let i = 0; i < cols; i++) norm += next[i] * next[i];
      norm = Math.sqrt(norm) || 1;
      for (let i = 0; i < cols; i++) vec[i] = next[i] / norm;
    }
    return vec;
  };

  const prevComponents: Float64Array[] = [];
  for (let which = 0; which < 2; which++) {
    const comp = component(which);
    prevComponents.push(comp);
    let min = Infinity;
    let max = -Infinity;
    const proj = new Float64Array(rows);
    for (let i = 0; i < rows; i++) {
      const off = i * cols;
      let s = 0;
      for (let j = 0; j < cols; j++) s += centred[off + j] * comp[j];
      proj[i] = s;
      if (s < min) min = s;
      if (s > max) max = s;
    }
    const span = max - min || 1;
    for (let i = 0; i < rows; i++) out[i * 2 + which] = (proj[i] - min) / span;
  }
  return out;
}
