import { Rng } from '../core/rng.ts';
import { generate } from '../model/sample.ts';
import type { BaseModel } from '../model/base.ts';
import type { Tokenizer } from '../data/tokenizer.ts';
import { Adam, DEFAULT_OPTIMIZER, type OptimizerConfig } from './optimizer.ts';

export interface TrainConfig {
  steps: number;
  batchSize: number;
  blockSize: number;
  evalEvery: number;
  evalBatches: number;
  sampleEvery: number;
  sampleLength: number;
  temperature: number;
  topK: number;
  seed: number;
  /** Fraction of the corpus used for training; the rest is validation. */
  trainSplit: number;
  /** Text used as the seed for sample printing. */
  prompt: string;
  optimizer: Partial<OptimizerConfig>;
}

export const DEFAULT_TRAIN: TrainConfig = {
  steps: 1000,
  batchSize: 32,
  blockSize: 32,
  evalEvery: 100,
  evalBatches: 8,
  sampleEvery: 250,
  sampleLength: 180,
  temperature: 0.8,
  topK: 40,
  seed: 1337,
  trainSplit: 0.9,
  prompt: '',
  optimizer: {},
};

export interface StepReport {
  step: number;
  loss: number;
  lr: number;
  gradNorm: number;
  tokensPerSec: number;
  elapsedMs: number;
}

const now = (): number => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/**
 * The training loop: sample a batch of random windows, forward, backward, Adam.
 *
 * It is exposed one step at a time (`stepOnce`) rather than as a blocking
 * `run()` so the browser worker can drip progress to the UI and stay
 * interruptible, while the CLI can just loop it flat out.
 */
export class Trainer {
  readonly model: BaseModel;
  readonly tokenizer: Tokenizer;
  readonly cfg: TrainConfig;
  readonly opt: Adam;
  readonly history: { step: number; loss: number; valLoss?: number }[] = [];
  step = 0;
  lastValLoss: number | null = null;
  lastSample = '';
  totalMs = 0;

  private train: Int32Array;
  private val: Int32Array;
  private batchTokens: Int32Array;
  private batchTargets: Int32Array;
  private evalTokens: Int32Array;
  private evalTargets: Int32Array;
  private rng: Rng;

  constructor(model: BaseModel, tokenizer: Tokenizer, data: Int32Array, cfg: Partial<TrainConfig> = {}) {
    const merged: TrainConfig = { ...DEFAULT_TRAIN, ...cfg, optimizer: { ...DEFAULT_OPTIMIZER, ...(cfg.optimizer ?? {}) } };
    if (model.config.blockSize !== merged.blockSize) {
      throw new Error(`model blockSize ${model.config.blockSize} != trainer blockSize ${merged.blockSize}`);
    }
    this.cfg = merged;
    this.model = model;
    this.tokenizer = tokenizer;
    this.rng = new Rng(merged.seed);
    const split = Math.max(merged.blockSize + 2, Math.floor(data.length * merged.trainSplit));
    this.train = data.slice(0, split);
    this.val = data.slice(split);
    const B = merged.batchSize;
    const T = merged.blockSize;
    this.batchTokens = new Int32Array(B * T);
    this.batchTargets = new Int32Array(B * T);
    this.evalTokens = new Int32Array(B * T);
    this.evalTargets = new Int32Array(B * T);
    this.opt = new Adam(model.params, merged.optimizer, merged.steps);
  }

  get trainTokens(): number {
    return this.train.length;
  }

  get valTokens(): number {
    return this.val.length;
  }

  private fillBatch(tokens: Int32Array, targets: Int32Array, source: Int32Array, seedOffset: number): void {
    const B = this.cfg.batchSize;
    const T = this.cfg.blockSize;
    const maxStart = source.length - T - 1;
    for (let b = 0; b < B; b++) {
      const start = Math.max(0, Math.min(maxStart, seedOffset + this.rng.int(Math.max(1, maxStart))));
      tokens.set(source.subarray(start, start + T), b * T);
      targets.set(source.subarray(start + 1, start + T + 1), b * T);
    }
  }

  /** One optimisation step. */
  stepOnce(): StepReport {
    const t0 = now();
    this.fillBatch(this.batchTokens, this.batchTargets, this.train, 0);
    this.model.zeroGrad();
    const loss = this.model.forward(this.batchTokens, this.batchTargets, this.cfg.batchSize, this.cfg.blockSize);
    this.model.backward();
    const gradNorm = this.opt.gradNorm();
    const lr = this.opt.step();
    const elapsed = now() - t0;
    this.step++;
    this.totalMs += elapsed;
    const tokensPerSec = (this.cfg.batchSize * this.cfg.blockSize) / (elapsed / 1000);
    this.history.push({ step: this.step, loss });
    return { step: this.step, loss, lr, gradNorm, tokensPerSec, elapsedMs: elapsed };
  }

  /** Mean loss over a fixed set of validation windows. */
  evaluate(): number {
    if (this.val.length < this.cfg.blockSize + 2) return Number.NaN;
    const B = this.cfg.batchSize;
    const T = this.cfg.blockSize;
    const maxStart = this.val.length - T - 1;
    let total = 0;
    for (let i = 0; i < this.cfg.evalBatches; i++) {
      for (let b = 0; b < B; b++) {
        const start = Math.floor(((i * B + b) / (this.cfg.evalBatches * B)) * maxStart);
        this.evalTokens.set(this.val.subarray(start, start + T), b * T);
        this.evalTargets.set(this.val.subarray(start + 1, start + T + 1), b * T);
      }
      total += this.model.forward(this.evalTokens, this.evalTargets, B, T);
    }
    const mean = total / this.cfg.evalBatches;
    this.lastValLoss = mean;
    const last = this.history[this.history.length - 1];
    if (last) last.valLoss = mean;
    return mean;
  }

  /**
   * Generate text from the current weights.
   * Note: this runs a forward pass, so callers that want the attention maps of
   * the latest training step must snapshot them first.
   */
  sampleText(prompt?: string, length?: number): string {
    const p = prompt ?? this.cfg.prompt;
    const encoded = Array.from(this.tokenizer.encode(p));
    const ids = encoded.length > 0 ? encoded : [0];
    const out = generate(this.model, ids, {
      maxNew: length ?? this.cfg.sampleLength,
      temperature: this.cfg.temperature,
      topK: this.cfg.topK,
      seed: this.cfg.seed + this.step,
    });
    this.lastSample = this.tokenizer.decode(out);
    return this.lastSample;
  }
}
