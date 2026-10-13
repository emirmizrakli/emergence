import type { Mat } from '../core/mat.ts';

/**
 * A trainable tensor plus the state Adam needs to update it.
 * Keeping first/second moments next to the parameters (instead of in a side
 * table) makes checkpointing a single structural clone.
 */
export interface Param {
  name: string;
  rows: number;
  cols: number;
  data: Float32Array;
  grad: Float32Array;
  /** Adam first moment */
  m: Float32Array;
  /** Adam second moment */
  v: Float32Array;
}

export function makeParam(name: string, rows: number, cols: number, init: (i: number, j: number) => number): Param {
  const size = rows * cols;
  const data = new Float32Array(size);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) data[i * cols + j] = init(i, j);
  }
  return {
    name,
    rows,
    cols,
    data,
    grad: new Float32Array(size),
    m: new Float32Array(size),
    v: new Float32Array(size),
  };
}

/** Zero-copy view of a parameter's values as a Mat. */
export function paramMat(p: Param): Mat {
  return { rows: p.rows, cols: p.cols, data: p.data };
}

/** Zero-copy view of a parameter's gradient as a Mat. */
export function paramGradMat(p: Param): Mat {
  return { rows: p.rows, cols: p.cols, data: p.grad };
}
