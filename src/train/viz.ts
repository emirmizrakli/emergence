import type { BaseModel } from '../model/base.ts';
import type { Tokenizer } from '../data/tokenizer.ts';
import { pca2 } from './pca.ts';

/**
 * Compact, structured-cloneable snapshots of the model's internals, taken
 * between training steps and shipped to the UI.
 *
 * Nothing here mutates the model, and nothing runs inside the training loop's
 * hot path — capture happens every N steps, so the visualisations cannot slow
 * training down noticeably.
 */

export interface AttentionSnapshot {
  layer: number;
  head: number;
  t: number;
  /** length t*t, rounded to 3 decimals to keep the payload small */
  weights: number[];
}

export interface EmbeddingSnapshot {
  /** one label per point */
  labels: string[];
  /** [x, y] in [0, 1] per point */
  points: [number, number][];
}

export interface VizSnapshot {
  step: number;
  loss: number;
  valLoss: number | null;
  lr: number;
  gradNorm: number;
  tokensPerSec: number;
  paramCount: number;
  attention: AttentionSnapshot[] | null;
  embedding: EmbeddingSnapshot | null;
  sample: string;
}

const round3 = (v: number): number => Math.round(v * 1000) / 1000;

export function captureAttention(model: BaseModel): AttentionSnapshot[] | null {
  if (model.attention.length === 0) return null;
  return model.attention.map((a) => ({
    layer: a.layer,
    head: a.head,
    t: a.t,
    weights: Array.from(a.weights, round3),
  }));
}

export function captureEmbedding(model: BaseModel, tokenizer: Tokenizer): EmbeddingSnapshot | null {
  const matrix = model.embeddingMatrix();
  if (!matrix || matrix.rows === 0) return null;
  const points = pca2(matrix);
  const labels: string[] = [];
  for (let i = 0; i < matrix.rows; i++) {
    const ch = tokenizer.chars[i];
    labels.push(ch === undefined ? `#${i}` : ch === ' ' ? '␣' : ch === '\n' ? '⏎' : ch);
  }
  const out: [number, number][] = [];
  for (let i = 0; i < matrix.rows; i++) out.push([round3(points[i * 2]), round3(points[i * 2 + 1])]);
  return { labels, points: out };
}

/** Mean attention entropy per head — a cheap "is it looking anywhere?" signal. */
export function attentionEntropy(weights: ArrayLike<number>, t: number): number {
  let total = 0;
  let rows = 0;
  for (let i = 0; i < t; i++) {
    let h = 0;
    for (let j = 0; j <= i; j++) {
      const p = weights[i * t + j];
      if (p > 1e-9) h -= p * Math.log(p);
    }
    total += h;
    rows++;
  }
  return rows > 0 ? total / rows : 0;
}
