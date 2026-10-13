import type { ModelConfig } from './base.ts';

/**
 * A closed-form FLOP budget for one forward + backward pass.
 *
 * Showing the real arithmetic cost (rather than a hand-wave) is part of the
 * point of this project: it makes the trade-offs between the three
 * architectures legible — the transformer's attention term is the only one
 * that grows quadratically in the context length, and you can watch that show
 * up as you drag `blockSize`.
 */
export interface FlopBreakdown {
  /** multiply-accumulate operations, counted once for forward and twice for backward */
  matmul: number;
  attention: number;
  normAndActivation: number;
  total: number;
  /** total FLOPs (2 x MACs) */
  flops: number;
  notes: string;
}

export function estimateFlops(config: ModelConfig, batchSize: number, blockSize: number): FlopBreakdown {
  const N = batchSize * blockSize;
  const C = config.nEmbd;
  const V = config.vocabSize;
  const L = config.nLayer;
  const T = blockSize;

  let matmul = 0;
  if (config.kind === 'gpt') {
    // qkv, attention output projection, mlp up, mlp down, and the tied head
    matmul += N * 3 * C * C;
    matmul += N * C * C;
    matmul += N * 4 * C * C;
    matmul += N * 4 * C * C;
    matmul += N * C * V;
    matmul *= L;
    matmul += N * C * V; // the head is outside the block loop
    matmul += V * C + T * C; // embedding lookups counted as one add each
  } else if (config.kind === 'mlp') {
    const ctx = Math.min(blockSize, 8);
    matmul += N * 4 * C * ctx * C;
    matmul += N * C * 4 * C;
    matmul += N * C * V;
  } else {
    matmul += N * V; // reading a row of the table
  }

  // attention scores and the weighted sum over values: 2 * T * C per position
  const attention = config.kind === 'gpt' ? L * N * 2 * T * C : 0;
  // five elementwise passes per normalisation, one for GELU
  const normAndActivation = config.kind === 'gpt' ? L * N * C * 6 + N * 4 * C * 2 : config.kind === 'mlp' ? N * 4 * C * 4 : N;

  const forwardMacs = matmul + attention + normAndActivation;
  const total = forwardMacs + 2 * (matmul + attention + normAndActivation);
  return {
    matmul,
    attention,
    normAndActivation,
    total,
    flops: 2 * total,
    notes: `${batchSize}x${blockSize} tokens, ${L} layer(s), C=${C}, ${config.nHead} head(s)`,
  };
}

export function formatFlops(flops: number): string {
  if (flops >= 1e12) return `${(flops / 1e12).toFixed(2)} TFLOP`;
  if (flops >= 1e9) return `${(flops / 1e9).toFixed(2)} GFLOP`;
  if (flops >= 1e6) return `${(flops / 1e6).toFixed(2)} MFLOP`;
  return `${flops.toFixed(0)} FLOP`;
}
