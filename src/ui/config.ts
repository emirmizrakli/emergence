import type { ModelKind } from '../model/base.ts';

/**
 * Sizes are chosen so that the page is useful at every step of the ladder:
 * `tiny` reaches readable text in seconds, `medium` is the one that makes the
 * fan spin. The bundled checkpoint is exactly `small`, so the page can start
 * from a model that already knows English.
 */
export interface Preset {
  label: string;
  nEmbd: number;
  nLayer: number;
  nHead: number;
  blockSize: number;
  batchSize: number;
  steps: number;
  approxParams: number;
}

export const PRESETS: Record<'tiny' | 'small' | 'medium', Preset> = {
  tiny: { label: 'tiny', nEmbd: 32, nLayer: 2, nHead: 4, blockSize: 32, batchSize: 24, steps: 8000, approxParams: 28_000 },
  small: { label: 'small', nEmbd: 64, nLayer: 4, nHead: 4, blockSize: 48, batchSize: 32, steps: 12000, approxParams: 204_000 },
  medium: { label: 'medium', nEmbd: 96, nLayer: 6, nHead: 6, blockSize: 64, batchSize: 40, steps: 16000, approxParams: 677_000 },
};

export type PresetName = keyof typeof PRESETS;

export const CORPORA: Record<string, { label: string; url: string; chars: string }> = {
  shakespeare: { label: 'tiny shakespeare (1.1 MB)', url: 'corpora/shakespeare.txt', chars: '~1.1M' },
};

export const WARM_START_URL = 'checkpoints/shakespeare-gpt.json';

export const DEFAULT_PROMPT = 'ROMEO: ';

export const ARCH_LABELS: Record<ModelKind, string> = {
  bigram: 'bigram table',
  mlp: 'fixed-window MLP',
  gpt: 'transformer (gpt)',
};
