import type { ModelConfig } from './base.ts';

/**
 * A human-readable description of the forward graph, with the real multiply-add
 * count attached to every node.
 *
 * This is the "tape" panel in the UI: not a decorative diagram, but the actual
 * sequence of tensor operations for the current configuration, sorted by cost.
 * It is generated from the same config the model is built from, so it cannot
 * drift away from what the code really does.
 */
export interface GraphRow {
  op: string;
  shape: string;
  macs: number;
  note?: string;
}

const s = (rows: number, cols: number): string => `[${rows},${cols}]`;

export function describeGraph(config: ModelConfig, batchSize: number, blockSize: number): GraphRow[] {
  const N = batchSize * blockSize;
  const C = config.nEmbd;
  const V = config.vocabSize;
  const T = blockSize;
  const L = config.nLayer;

  if (config.kind === 'bigram') {
    return [
      { op: 'lookup', shape: s(N, V), macs: 0, note: 'row of the table = logits' },
      { op: 'cross-entropy', shape: s(N, V), macs: N * V * 3, note: 'softmax + log' },
      { op: 'table.grad', shape: s(V, V), macs: N * V, note: 'one-hot scatter' },
    ];
  }

  if (config.kind === 'mlp') {
    const ctx = Math.min(blockSize, 8);
    return [
      { op: `gather ${ctx} embeddings`, shape: s(N, ctx * C), macs: 0 },
      { op: 'fc1 = x @ W1ᵀ + b1', shape: s(N, 4 * C), macs: N * 4 * C * ctx * C },
      { op: 'gelu', shape: s(N, 4 * C), macs: N * 4 * C },
      { op: 'fc2 = h @ W2ᵀ + b2', shape: s(N, C), macs: N * C * 4 * C },
      { op: 'logits = fc2 @ wteᵀ  (tied)', shape: s(N, V), macs: N * C * V },
      { op: 'cross-entropy', shape: s(N, V), macs: N * V * 3 },
    ];
  }

  const rows: GraphRow[] = [
    { op: 'tokens + positions → x', shape: s(N, C), macs: N * C },
  ];
  const perLayer: GraphRow[] = [
    { op: 'ln1 = LayerNorm(x)', shape: s(N, C), macs: N * C * 3 },
    { op: 'qkv = ln1 @ Wqkvᵀ', shape: s(N, 3 * C), macs: N * 3 * C * C },
    { op: 'scores = q kᵀ / √d', shape: s(config.nHead * N, T), macs: N * T * C, note: 'causal mask' },
    { op: 'softmax(scores)', shape: s(config.nHead * N, T), macs: N * T * 3 },
    { op: 'att = softmax @ v', shape: s(N, C), macs: N * T * C },
    { op: 'attn = att @ Wprojᵀ', shape: s(N, C), macs: N * C * C },
    { op: 'x += attn  (residual)', shape: s(N, C), macs: N * C },
    { op: 'ln2 = LayerNorm(x)', shape: s(N, C), macs: N * C * 3 },
    { op: 'h = ln2 @ Wfcᵀ + b', shape: s(N, 4 * C), macs: N * 4 * C * C },
    { op: 'gelu(h)', shape: s(N, 4 * C), macs: N * 4 * C },
    { op: 'mlp = h @ Wdownᵀ + b', shape: s(N, C), macs: N * C * 4 * C },
    { op: 'x += mlp  (residual)', shape: s(N, C), macs: N * C },
  ];
  for (let l = 0; l < L; l++) {
    for (const row of perLayer) {
      rows.push(l === 0 ? row : { ...row, op: `h${l}.${row.op}`, note: row.note });
    }
  }
  rows.push(
    { op: 'lnf = LayerNorm(x)', shape: s(N, C), macs: N * C * 3 },
    { op: 'logits = lnf @ wteᵀ  (tied)', shape: s(N, V), macs: N * C * V },
    { op: 'cross-entropy', shape: s(N, V), macs: N * V * 3 },
  );
  return rows;
}

export function totalGraphMacs(rows: GraphRow[]): number {
  let m = 0;
  for (const r of rows) m += r.macs;
  return m;
}
