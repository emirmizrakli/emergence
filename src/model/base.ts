import type { Mat } from '../core/mat.ts';
import type { Param } from './param.ts';

export type ModelKind = 'bigram' | 'mlp' | 'gpt';

export interface ModelConfig {
  kind: ModelKind;
  vocabSize: number;
  /** Maximum context length the model conditions on. */
  blockSize: number;
  /** Embedding width (C). */
  nEmbd: number;
  /** Transformer blocks (gpt) or hidden layers (mlp). */
  nLayer: number;
  /** Attention heads (gpt). */
  nHead: number;
  seed: number;
}

/** Attention weights for one (layer, head) pair, first sequence in the batch. */
export interface AttentionMap {
  layer: number;
  head: number;
  /** length T*T, row-major; row t attends over columns 0..t */
  weights: Float32Array;
  t: number;
}

/**
 * Common surface for every architecture in the ladder (bigram → MLP → GPT).
 * The trainer, the CLI and the web worker all talk to models through this.
 */
export abstract class BaseModel {
  readonly config: ModelConfig;
  readonly params: Param[] = [];
  /** Logits of the most recent forward pass, [B*T, vocabSize]. */
  logits: Mat = { rows: 0, cols: 0, data: new Float32Array(0) };
  /** Loss returned by the most recent forward pass. */
  lastLoss = 0;
  /** Attention maps from the most recent forward pass (gpt only). */
  attention: AttentionMap[] = [];

  /** Batch from the most recent forward pass, so backward() needs no arguments. */
  protected tokens: Int32Array = new Int32Array(0);
  protected targets: Int32Array = new Int32Array(0);
  /** Row-wise softmax of the most recent logits; reused by the cross-entropy gradient. */
  protected probs: Mat = { rows: 0, cols: 0, data: new Float32Array(0) };

  constructor(config: ModelConfig) {
    this.config = config;
  }

  /** Copy the incoming batch into stable storage (callers may reuse their buffers). */
  protected cacheBatch(tokens: Int32Array, targets: Int32Array): void {
    if (this.tokens.length !== tokens.length) {
      this.tokens = new Int32Array(tokens.length);
      this.targets = new Int32Array(targets.length);
    }
    this.tokens.set(tokens);
    this.targets.set(targets);
  }

  abstract forward(tokens: Int32Array, targets: Int32Array, B: number, T: number): number;
  abstract backward(): void;

  zeroGrad(): void {
    for (const p of this.params) p.grad.fill(0);
  }

  numParams(): number {
    let n = 0;
    for (const p of this.params) n += p.rows * p.cols;
    return n;
  }

  /** Matrix whose rows are token embeddings, for the "space" visualiser. */
  embeddingMatrix(): Mat | null {
    return null;
  }

  /** Attention weight matrix for a given (layer, head), if this model has any. */
  attentionFor(layer: number, head: number): AttentionMap | null {
    return this.attention.find((a) => a.layer === layer && a.head === head) ?? null;
  }

  /** Token ids of the batch in the most recent forward pass (labels the UI's attention panel). */
  lastBatchTokens(): Int32Array {
    return this.tokens;
  }

  get numLayers(): number {
    return this.config.nLayer;
  }

  get numHeads(): number {
    return this.config.kind === 'gpt' ? this.config.nHead : 0;
  }

  protected register(p: Param): Param {
    this.params.push(p);
    return p;
  }
}
