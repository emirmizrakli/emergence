import {
  addBias,
  copyInto,
  crossEntropy,
  crossEntropyBackward,
  geluBackwardInPlace,
  geluInPlace,
  mat,
  matmulNN,
  matmulNT,
  matmulTN,
  sumRowsInto,
  type Mat,
} from '../core/mat.ts';
import { Rng } from '../core/rng.ts';
import { BaseModel, type ModelConfig } from './base.ts';
import { makeParam, paramGradMat, paramMat, type Param } from './param.ts';

/**
 * The middle rung: a Bengio-style fixed-window MLP language model.
 *
 * Instead of relating every position to every other one (attention), it
 * concatenates the embeddings of the previous `context` tokens and pushes them
 * through one hidden layer with a GELU non-linearity. It learns local syntax
 * far better than the bigram table, but it cannot see past its window — a
 * limitation the transformer's attention heatmaps make obvious.
 */
export class MlpModel extends BaseModel {
  private wte: Param;
  private fc1W: Param;
  private fc1B: Param;
  private fc2W: Param;
  private fc2B: Param;
  readonly context: number;

  private xin!: Mat;
  private fc1pre!: Mat;
  private h1!: Mat;
  private fc2!: Mat;
  private dlogits!: Mat;
  private dxin!: Mat;
  private dfc2!: Mat;
  private dfc1pre!: Mat;
  private bufN = -1;
  private bufT = 1;

  constructor(config: ModelConfig) {
    super(config);
    const { vocabSize: V, nEmbd: C, seed, blockSize } = config;
    this.context = Math.max(1, Math.min(blockSize, 8));
    const rng = new Rng(seed);
    const std = 0.02;
    this.wte = this.register(makeParam('wte', V, C, () => rng.normal(std)));
    this.fc1W = this.register(makeParam('mlp.fc1.weight', 4 * C, this.context * C, () => rng.normal(std)));
    this.fc1B = this.register(makeParam('mlp.fc1.bias', 4 * C, 1, () => 0));
    this.fc2W = this.register(makeParam('mlp.fc2.weight', C, 4 * C, () => rng.normal(std)));
    this.fc2B = this.register(makeParam('mlp.fc2.bias', C, 1, () => 0));
  }

  private ensure(N: number): void {
    if (this.bufN === N) return;
    this.bufN = N;
    const C = this.config.nEmbd;
    const F = 4 * C;
    const win = this.context * C;
    const V = this.config.vocabSize;
    this.xin = mat(N, win);
    this.fc1pre = mat(N, F);
    this.h1 = mat(N, F);
    this.fc2 = mat(N, C);
    this.dlogits = mat(N, V);
    this.dxin = mat(N, win);
    this.dfc2 = mat(N, C);
    this.dfc1pre = mat(N, F);
    this.logits = mat(N, V);
    this.probs = mat(N, V);
  }

  forward(tokens: Int32Array, targets: Int32Array, B: number, T: number): number {
    this.cacheBatch(tokens, targets);
    this.bufT = T;
    const N = B * T;
    this.ensure(N);
    const C = this.config.nEmbd;
    const win = this.context * C;
    const wte = this.wte.data;

    // xin[n] = concatenated embeddings of the previous `context` tokens
    this.xin.data.fill(0);
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const n = b * T + t;
        for (let k = 0; k < this.context; k++) {
          const src = t - (this.context - 1) + k;
          if (src < 0) continue;
          const from = this.tokens[b * T + src] * C;
          const to = n * win + k * C;
          for (let c = 0; c < C; c++) this.xin.data[to + c] = wte[from + c];
        }
      }
    }

    matmulNT(this.xin, paramMat(this.fc1W), this.fc1pre);
    addBias(this.fc1pre, this.fc1B.data);
    copyInto(this.h1, this.fc1pre);
    geluInPlace(this.h1);
    matmulNT(this.h1, paramMat(this.fc2W), this.fc2);
    addBias(this.fc2, this.fc2B.data);
    matmulNT(this.fc2, paramMat(this.wte), this.logits);
    this.lastLoss = crossEntropy(this.logits, this.targets, this.probs);
    return this.lastLoss;
  }

  backward(): void {
    const C = this.config.nEmbd;
    const win = this.context * C;
    const T = this.bufT;
    const N = this.bufN;

    crossEntropyBackward(this.dlogits, this.probs, this.targets);
    matmulNN(this.dlogits, paramMat(this.wte), this.dfc2);
    matmulTN(this.dlogits, this.fc2, paramGradMat(this.wte), true);
    matmulNN(this.dfc2, paramMat(this.fc2W), this.dfc1pre);
    matmulTN(this.dfc2, this.h1, paramGradMat(this.fc2W), true);
    sumRowsInto(this.fc2B.grad, this.dfc2);
    geluBackwardInPlace(this.dfc1pre, this.fc1pre.data);
    matmulNN(this.dfc1pre, paramMat(this.fc1W), this.dxin);
    matmulTN(this.dfc1pre, this.xin, paramGradMat(this.fc1W), true);
    sumRowsInto(this.fc1B.grad, this.dfc1pre);

    // scatter window gradients back into the embedding table
    const wteGrad = this.wte.grad;
    for (let n = 0; n < N; n++) {
      const t = n % T;
      const base = n - t;
      for (let k = 0; k < this.context; k++) {
        const src = t - (this.context - 1) + k;
        if (src < 0) continue;
        const to = this.tokens[base + src] * C;
        const from = n * win + k * C;
        for (let c = 0; c < C; c++) wteGrad[to + c] += this.dxin.data[from + c];
      }
    }
  }

  override embeddingMatrix(): Mat | null {
    return paramMat(this.wte);
  }
}
