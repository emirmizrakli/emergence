/**
 * A tiny reverse-mode autodiff engine, kept deliberately separate from the
 * hand-written kernels in mat.ts.
 *
 * Why two engines?
 *  1. The hand-written path is fast enough to train a real model in a browser
 *     tab, but its gradients are derived by hand — easy to get subtly wrong.
 *  2. This tape-based engine derives gradients automatically from the forward
 *     pass, so it is an *independent* oracle: src/core/autograd.test.ts checks
 *     both engines against each other and against finite differences.
 *
 * It is also what feeds the "tape" panel in the UI: every op is visible.
 */

export type TapeNode = {
  op: string;
  id: number;
  outShape: [number, number];
};

type Backward = (grad: Float32Array) => void;

export class Var {
  static counter = 0;
  readonly id: number;
  readonly rows: number;
  readonly cols: number;
  data: Float32Array;
  grad: Float32Array;
  parents: Var[] = [];
  /** Pushes this node's gradient into its parents (set by link()). */
  propagate: Backward = () => {};
  op: string;

  constructor(data: Float32Array | number[], rows: number, cols: number, op = 'leaf') {
    this.id = Var.counter++;
    this.rows = rows;
    this.cols = cols;
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.grad = new Float32Array(rows * cols);
    this.op = op;
  }

  get size(): number {
    return this.rows * this.cols;
  }

  /** Register how to push gradient into the parents of this node. */
  private link(parents: Var[], backward: Backward, op: string): this {
    this.parents = parents;
    this.propagate = backward;
    this.op = op;
    return this;
  }

  zeroGrad(): void {
    this.grad.fill(0);
  }

  /** Elementwise add of a same-shaped tensor. */
  add(other: Var): Var {
    const { rows, cols } = this;
    const out = new Float32Array(rows * cols);
    for (let i = 0; i < out.length; i++) out[i] = this.data[i] + other.data[i];
    const res = new Var(out, rows, cols);
    return res.link([this, other], (g) => {
      for (let i = 0; i < g.length; i++) {
        this.grad[i] += g[i];
        other.grad[i] += g[i];
      }
    }, 'add');
  }

  /** this @ other  (2-D matmul) */
  matmul(other: Var): Var {
    if (this.cols !== other.rows) throw new Error('matmul shape mismatch');
    const N = this.rows;
    const K = this.cols;
    const M = other.cols;
    const out = new Float32Array(N * M);
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < K; k++) {
        const a = this.data[i * K + k];
        if (a === 0) continue;
        for (let j = 0; j < M; j++) out[i * M + j] += a * other.data[k * M + j];
      }
    }
    const res = new Var(out, N, M);
    return res.link([this, other], (g) => {
      for (let i = 0; i < N; i++) {
        for (let k = 0; k < K; k++) {
          let s = 0;
          for (let j = 0; j < M; j++) s += g[i * M + j] * other.data[k * M + j];
          this.grad[i * K + k] += s;
        }
      }
      for (let k = 0; k < K; k++) {
        for (let j = 0; j < M; j++) {
          let s = 0;
          for (let i = 0; i < N; i++) s += this.data[i * K + k] * g[i * M + j];
          other.grad[k * M + j] += s;
        }
      }
    }, 'matmul');
  }

  /** Row-wise softmax (exp-normalised over each row). */
  softmaxRows(): Var {
    const { rows, cols } = this;
    const out = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      const off = i * cols;
      let max = -Infinity;
      for (let j = 0; j < cols; j++) if (this.data[off + j] > max) max = this.data[off + j];
      let sum = 0;
      for (let j = 0; j < cols; j++) {
        const v = Math.exp(this.data[off + j] - max);
        out[off + j] = v;
        sum += v;
      }
      for (let j = 0; j < cols; j++) out[off + j] /= sum;
    }
    const res = new Var(out, rows, cols);
    return res.link([this], (g) => {
      for (let i = 0; i < rows; i++) {
        const off = i * cols;
        let dot = 0;
        for (let j = 0; j < cols; j++) dot += g[off + j] * out[off + j];
        for (let j = 0; j < cols; j++) this.grad[off + j] += out[off + j] * (g[off + j] - dot);
      }
    }, 'softmax');
  }

  /** Row-wise layer normalisation with learnable scale/shift. */
  layernorm(scale: Var, shift: Var, eps = 1e-5): Var {
    const { rows, cols } = this;
    const out = new Float32Array(rows * cols);
    const mean = new Float32Array(rows);
    const rstd = new Float32Array(rows);
    for (let i = 0; i < rows; i++) {
      const off = i * cols;
      let m = 0;
      for (let j = 0; j < cols; j++) m += this.data[off + j];
      m /= cols;
      let v = 0;
      for (let j = 0; j < cols; j++) {
        const d = this.data[off + j] - m;
        v += d * d;
      }
      v /= cols;
      const r = 1 / Math.sqrt(v + eps);
      mean[i] = m;
      rstd[i] = r;
      for (let j = 0; j < cols; j++) out[off + j] = (this.data[off + j] - m) * r * scale.data[j] + shift.data[j];
    }
    const res = new Var(out, rows, cols);
    return res.link([this, scale, shift], (g) => {
      for (let i = 0; i < rows; i++) {
        const off = i * cols;
        const m = mean[i];
        const r = rstd[i];
        let s1 = 0;
        let s2 = 0;
        for (let j = 0; j < cols; j++) {
          const xHat = (this.data[off + j] - m) * r;
          const dg = g[off + j] * scale.data[j];
          s1 += dg;
          s2 += dg * xHat;
        }
        s1 /= cols;
        s2 /= cols;
        for (let j = 0; j < cols; j++) {
          const xHat = (this.data[off + j] - m) * r;
          const dg = g[off + j] * scale.data[j];
          this.grad[off + j] += r * (dg - s1 - xHat * s2);
          scale.grad[j] += g[off + j] * xHat;
          shift.grad[j] += g[off + j];
        }
      }
    }, 'layernorm');
  }

  /** Elementwise tanh-approximation GELU. */
  gelu(): Var {
    const n = this.size;
    const out = new Float32Array(n);
    const c = 0.7978845608028654;
    for (let i = 0; i < n; i++) {
      const x = this.data[i];
      out[i] = 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
    }
    const res = new Var(out, this.rows, this.cols);
    return res.link([this], (g) => {
      for (let i = 0; i < n; i++) {
        const x = this.data[i];
        const inner = c * (x + 0.044715 * x * x * x);
        const t = Math.tanh(inner);
        const sech2 = 1 - t * t;
        const din = c * (1 + 3 * 0.044715 * x * x);
        this.grad[i] += g[i] * (0.5 * (1 + t) + 0.5 * x * sech2 * din);
      }
    }, 'gelu');
  }

  /** Broadcast a [1, C] bias across rows. */
  addBias(bias: Var): Var {
    const { rows, cols } = this;
    const out = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      const off = i * cols;
      for (let j = 0; j < cols; j++) out[off + j] = this.data[off + j] + bias.data[j];
    }
    const res = new Var(out, rows, cols);
    return res.link([this, bias], (g) => {
      for (let i = 0; i < rows; i++) {
        const off = i * cols;
        for (let j = 0; j < cols; j++) {
          this.grad[off + j] += g[off + j];
          bias.grad[j] += g[off + j];
        }
      }
    }, 'addBias');
  }

  get T(): Var {
    const { rows, cols } = this;
    const out = new Float32Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) out[j * rows + i] = this.data[i * cols + j];
    }
    const res = new Var(out, cols, rows);
    return res.link([this], (g) => {
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) this.grad[i * cols + j] += g[j * rows + i];
      }
    }, 'transpose');
  }

  /** Sum of elementwise squares, as a [1,1] tensor. Useful as a scratch loss. */
  meanSquare(): Var {
    const n = this.size;
    let s = 0;
    for (let i = 0; i < n; i++) s += this.data[i] * this.data[i];
    const res = new Var([s / n], 1, 1);
    return res.link([this], (g) => {
      const k = (g[0] * 2) / n;
      for (let i = 0; i < n; i++) this.grad[i] += k * this.data[i];
    }, 'meanSquare');
  }

  /**
   * Backpropagate from this node (assumed scalar). Topological order is
   * recovered by walking parents and ordering by descending id, which is a
   * valid order for a graph built strictly forwards.
   */
  backward(seed = 1): void {
    this.grad[0] += seed;
    const topo: Var[] = [];
    const seen = new Set<number>();
    const visit = (node: Var): void => {
      if (seen.has(node.id)) return;
      seen.add(node.id);
      for (const p of node.parents) visit(p);
      topo.push(node);
    };
    visit(this);
    for (let i = topo.length - 1; i >= 0; i--) {
      const node = topo[i];
      node.propagate(node.grad);
    }
  }

  /** A snapshot of this node, for the UI's tape panel. */
  get tape(): TapeNode {
    return { op: this.op, id: this.id, outShape: [this.rows, this.cols] };
  }
}

/** Convenience: build a Var from a Mat-like object. */
export function varOf(rows: number, cols: number, fill: (i: number, j: number) => number): Var {
  const data = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) data[i * cols + j] = fill(i, j);
  }
  return new Var(data, rows, cols);
}
