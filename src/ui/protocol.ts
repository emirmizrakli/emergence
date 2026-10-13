import type { ModelKind, ModelConfig } from '../model/base.ts';
import type { AttentionSnapshot, EmbeddingSnapshot } from '../train/viz.ts';
import type { GraphRow } from '../model/graph.ts';
import type { PresetName } from './config.ts';

/** Messages the page sends to the training worker. */

export interface StartRequest {
  type: 'start';
  arch: ModelKind;
  preset: PresetName;
  lr: number;
  corpusText: string;
  corpusName: string;
  /** checkpoint JSON, or null to start from random noise */
  warmStart: string | null;
  prompt: string;
}

export type MainToWorker =
  | StartRequest
  | { type: 'pause' }
  | { type: 'resume' }
  | { type: 'inspect'; prompt: string }
  | { type: 'attention'; layer: number; head: number }
  | { type: 'export'; meta: string };

/** Messages the worker sends back. */

export interface Metrics {
  step: number;
  plannedSteps: number;
  loss: number;
  smoothLoss: number;
  valLoss: number | null;
  lr: number;
  gradNorm: number;
  tokensPerSec: number;
  stepMs: number;
  elapsedMs: number;
  paramCount: number;
  arch: ModelKind;
  config: ModelConfig;
  vocabSize: number;
  context: number;
  flopsPerStep: number;
  flopsLabel: string;
  running: boolean;
  finished: boolean;
  warmStarted: boolean;
  corpusName: string;
  corpusChars: number;
}

export interface NextRow {
  char: string;
  prob: number;
}

export interface GradRow {
  name: string;
  gradNorm: number;
}

export type WorkerToMain =
  | { type: 'started'; metrics: Metrics; graph: GraphRow[]; heads: { layer: number; head: number }[] }
  | {
      type: 'update';
      metrics: Metrics;
      attention: AttentionSnapshot | null;
      embedding: EmbeddingSnapshot | null;
      next?: NextRow[];
      grads?: GradRow[];
      /** characters of the batch behind the attention map, for axis labels */
      chars?: string[];
      sampleChunk?: string;
      sampleReset?: boolean;
      sampleStep?: number;
    }
  | { type: 'error'; message: string }
  | { type: 'exported'; json: string; filename: string };
