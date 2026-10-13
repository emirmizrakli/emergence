import type { ModelConfig } from '../model/base.ts';
import type { BaseModel } from '../model/base.ts';
import { createModel } from '../model/factory.ts';
import { Tokenizer } from '../data/tokenizer.ts';

/**
 * Checkpoints: the trained weights, the tokenizer and the exact config, as
 * plain JSON.
 *
 * A checkpoint is what makes the browser demo instant — the page loads a model
 * that has already learned something and keeps training from there — and what
 * makes a CLI run reproducible: same checkpoint, same samples.
 */

export interface CheckpointMeta {
  name?: string;
  corpus?: string;
  steps?: number;
  loss?: number;
  valLoss?: number | null;
  createdAt?: string;
  notes?: string;
}

export interface Checkpoint {
  format: 'emergence-checkpoint';
  version: 1;
  config: ModelConfig;
  tokenizer: { chars: string[] };
  meta: CheckpointMeta;
  params: Record<string, { shape: [number, number]; data: string }>;
}

export function toBase64(bytes: Uint8Array): string {
  const g = globalThis as unknown as { Buffer?: { from(b: Uint8Array): { toString(enc: string): string } }; btoa?: (s: string) => string };
  if (g.Buffer) return g.Buffer.from(bytes).toString('base64');
  let s = '';
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    s += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return g.btoa ? g.btoa(s) : s;
}

export function fromBase64(b64: string): Uint8Array {
  const g = globalThis as unknown as { Buffer?: { from(s: string, enc: string): Uint8Array }; atob?: (s: string) => string };
  if (g.Buffer) return new Uint8Array(g.Buffer.from(b64, 'base64'));
  const bin = g.atob ? g.atob(b64) : b64;
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

export function encodeCheckpoint(model: BaseModel, tokenizer: Tokenizer, meta: CheckpointMeta = {}): Checkpoint {
  const params: Checkpoint['params'] = {};
  for (const p of model.params) {
    const bytes = new Uint8Array(p.data.buffer, p.data.byteOffset, p.data.byteLength);
    params[p.name] = { shape: [p.rows, p.cols], data: toBase64(bytes) };
  }
  return {
    format: 'emergence-checkpoint',
    version: 1,
    config: model.config,
    tokenizer: tokenizer.toJSON(),
    meta: { createdAt: new Date().toISOString(), ...meta },
    params,
  };
}

export function encodeCheckpointString(model: BaseModel, tokenizer: Tokenizer, meta: CheckpointMeta = {}): string {
  return JSON.stringify(encodeCheckpoint(model, tokenizer, meta));
}

export function decodeCheckpoint(raw: string | Checkpoint): { model: BaseModel; tokenizer: Tokenizer; meta: CheckpointMeta } {
  const ckpt: Checkpoint = typeof raw === 'string' ? (JSON.parse(raw) as Checkpoint) : raw;
  if (ckpt.format !== 'emergence-checkpoint') throw new Error(`unknown checkpoint format: ${String(ckpt.format)}`);
  const model = createModel(ckpt.config);
  for (const p of model.params) {
    const entry = ckpt.params[p.name];
    if (!entry) throw new Error(`checkpoint is missing parameter ${p.name}`);
    const bytes = fromBase64(entry.data);
    if (bytes.byteLength !== p.data.byteLength) {
      throw new Error(`parameter ${p.name} has ${bytes.byteLength} bytes, expected ${p.data.byteLength}`);
    }
    p.data.set(new Float32Array(bytes.buffer, bytes.byteOffset, p.data.length));
  }
  const tokenizer = Tokenizer.fromJSON(ckpt.tokenizer);
  return { model, tokenizer, meta: ckpt.meta ?? {} };
}
