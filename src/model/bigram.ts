import { mat, crossEntropy, crossEntropyBackward, type Mat } from '../core/mat.ts';
import { Rng } from '../core/rng.ts';
import { BaseModel, type ModelConfig } from './base.ts';
import { makeParam, paramMat, type Param } from './param.ts';

/**
 * The floor of the ladder: P(next token) read straight off a table.
 *
 * There is no hidden state at all — each prediction sees only the current
 * token — so it converges almost instantly to whatever the bigram statistics
 * of the corpus allow, and then stops. Watching it plateau next to the MLP
 * and the transformer is the point.
 */
export class BigramModel extends BaseModel {
  private W: Param;
  private dlogits: Mat = mat(0, 0);

  constructor(config: ModelConfig) {
    super(config);
    const rng = new Rng(config.seed);
    // Small init so the initial loss lands near ln(V).
    this.W = this.register(makeParam('bigram.table', config.vocabSize, config.vocabSize, () => rng.normal(0.05)));
  }

  forward(tokens: Int32Array, targets: Int32Array, B: number, T: number): number {
    this.cacheBatch(tokens, targets);
    const V = this.config.vocabSize;
    const N = B * T;
    if (this.logits.rows !== N || this.logits.cols !== V) {
      this.logits = mat(N, V);
      this.probs = mat(N, V);
      this.dlogits = mat(N, V);
    }
    const logits = this.logits.data;
    const W = this.W.data;
    for (let n = 0; n < N; n++) {
      const src = this.tokens[n] * V;
      logits.set(W.subarray(src, src + V), n * V);
    }
    this.lastLoss = crossEntropy(this.logits, this.targets, this.probs);
    return this.lastLoss;
  }

  backward(): void {
    const V = this.config.vocabSize;
    const N = this.logits.rows;
    crossEntropyBackward(this.dlogits, this.probs, this.targets);
    const dW = this.W.grad;
    const d = this.dlogits.data;
    for (let n = 0; n < N; n++) {
      const src = this.tokens[n] * V;
      const dst = n * V;
      for (let j = 0; j < V; j++) dW[src + j] += d[dst + j];
    }
  }

  override embeddingMatrix(): Mat | null {
    return paramMat(this.W);
  }
}
