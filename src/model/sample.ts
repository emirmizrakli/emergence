import type { BaseModel } from './base.ts';
import { Rng } from '../core/rng.ts';

export interface GenerateOptions {
  maxNew: number;
  temperature?: number;
  topK?: number;
  seed?: number;
}

/** Pick the next token id from a row of logits, with temperature and top-k. */
export function sampleFromLogits(
  logits: Float32Array,
  offset: number,
  vocabSize: number,
  temperature: number,
  topK: number,
  rng: Rng,
): number {
  if (temperature <= 0) {
    // greedy
    let best = 0;
    let bestV = -Infinity;
    for (let i = 0; i < vocabSize; i++) {
      const v = logits[offset + i];
      if (v > bestV) {
        bestV = v;
        best = i;
      }
    }
    return best;
  }

  const k = topK > 0 && topK < vocabSize ? topK : vocabSize;
  // Partial selection of the k largest logits (values only, ids tracked).
  const topVals = new Float32Array(k).fill(-Infinity);
  const topIds = new Int32Array(k).fill(-1);
  for (let i = 0; i < vocabSize; i++) {
    const v = logits[offset + i];
    if (v <= topVals[k - 1]) continue;
    let j = k - 1;
    while (j > 0 && topVals[j - 1] < v) {
      topVals[j] = topVals[j - 1];
      topIds[j] = topIds[j - 1];
      j--;
    }
    topVals[j] = v;
    topIds[j] = i;
  }

  let max = -Infinity;
  for (let i = 0; i < k; i++) if (topVals[i] > max) max = topVals[i];
  let sum = 0;
  const weights = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    const w = Math.exp((topVals[i] - max) / temperature);
    weights[i] = w;
    sum += w;
  }
  let r = rng.next() * sum;
  for (let i = 0; i < k; i++) {
    r -= weights[i];
    if (r <= 0) return topIds[i];
  }
  return topIds[k - 1];
}

/**
 * Autoregressive generation, one step at a time.
 *
 * Two deliberate choices:
 *
 * 1. No KV cache. Every new token re-runs the forward pass over the whole
 *    window, so the inference path is identical to the training path — what you
 *    see sampled is exactly the network you just trained, with no second
 *    implementation that could drift away from it.
 * 2. The window is always exactly `blockSize` long: live tokens on the left,
 *    padding on the right. Because attention is causal, the prediction at row
 *    `n-1` cannot be affected by anything to its right, so the padding is free
 *    — and holding one fixed shape means generation reuses the same activation
 *    buffers as training instead of allocating a fresh set for every token.
 *
 * `advance()` exists so the browser worker can emit a few characters per frame
 * instead of blocking for a whole paragraph.
 */
export class Generator {
  readonly blockSize: number;
  readonly vocabSize: number;
  readonly maxNew: number;
  produced = 0;

  private model: BaseModel;
  private window: Int32Array;
  private filled = 0;
  private tokens: Int32Array;
  private targets: Int32Array;
  private rng: Rng;
  private temperature: number;
  private topK: number;

  constructor(model: BaseModel, promptIds: number[], opts: GenerateOptions) {
    if (promptIds.length === 0) throw new Error('prompt must contain at least one token');
    const { maxNew, temperature = 0.8, topK = 40, seed = 1234 } = opts;
    this.model = model;
    this.blockSize = model.config.blockSize;
    this.vocabSize = model.config.vocabSize;
    this.maxNew = maxNew;
    this.temperature = temperature;
    this.topK = topK;
    this.rng = new Rng(seed);
    this.tokens = new Int32Array(this.blockSize);
    this.targets = new Int32Array(this.blockSize);
    this.window = new Int32Array(this.blockSize);
    const tail = promptIds.length > this.blockSize ? promptIds.slice(promptIds.length - this.blockSize) : promptIds;
    for (let i = 0; i < tail.length; i++) this.window[i] = tail[i];
    this.filled = tail.length;
  }

  /** Produce up to `k` more token ids. */
  advance(k: number): number[] {
    const out: number[] = [];
    for (let i = 0; i < k && this.produced < this.maxNew; i++) {
      this.tokens.fill(0);
      this.tokens.set(this.window.subarray(0, this.filled), 0);
      this.model.forward(this.tokens, this.targets, 1, this.blockSize);
      const offset = (this.filled - 1) * this.vocabSize;
      const next = sampleFromLogits(this.model.logits.data, offset, this.vocabSize, this.temperature, this.topK, this.rng);
      out.push(next);
      this.produced++;
      if (this.filled < this.blockSize) {
        this.window[this.filled++] = next;
      } else {
        this.window.copyWithin(0, 1);
        this.window[this.blockSize - 1] = next;
      }
    }
    return out;
  }

  /** Convenience: generate everything at once. */
  run(): number[] {
    return this.advance(this.maxNew);
  }
}

export function generate(model: BaseModel, promptIds: number[], opts: GenerateOptions): number[] {
  return new Generator(model, promptIds, opts).run();
}
