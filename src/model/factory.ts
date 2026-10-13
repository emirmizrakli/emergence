import { Rng } from '../core/rng.ts';
import { BigramModel } from './bigram.ts';
import { GptModel } from './gpt.ts';
import { MlpModel } from './mlp.ts';
import { BigramModel as _b } from './bigram.ts';
import type { BaseModel, ModelConfig, ModelKind } from './base.ts';

void _b;

/** Sensible defaults per architecture so the UI can offer one-click presets. */
export const MODEL_PRESETS: Record<ModelKind, Partial<ModelConfig>> = {
  bigram: { nEmbd: 32, nLayer: 1, nHead: 1, blockSize: 32 },
  mlp: { nEmbd: 64, nLayer: 1, nHead: 1, blockSize: 32 },
  gpt: { nEmbd: 64, nLayer: 2, nHead: 4, blockSize: 32 },
};

export function createModel(config: ModelConfig): BaseModel {
  switch (config.kind) {
    case 'bigram':
      return new BigramModel(config);
    case 'mlp':
      return new MlpModel(config);
    case 'gpt':
      return new GptModel(config);
    default: {
      const never: never = config.kind;
      throw new Error(`unknown model kind: ${String(never)}`);
    }
  }
}

export function configFor(kind: ModelKind, vocabSize: number, overrides: Partial<ModelConfig> = {}): ModelConfig {
  const preset = MODEL_PRESETS[kind];
  return {
    kind,
    vocabSize,
    blockSize: overrides.blockSize ?? preset.blockSize ?? 32,
    nEmbd: overrides.nEmbd ?? preset.nEmbd ?? 64,
    nLayer: overrides.nLayer ?? preset.nLayer ?? 1,
    nHead: overrides.nHead ?? preset.nHead ?? 1,
    seed: overrides.seed ?? 1337,
  };
}

/** Convenience for tests: a tiny deterministic model. */
export function tinyConfig(kind: ModelKind, vocabSize: number, seed = 1): ModelConfig {
  return configFor(kind, vocabSize, { blockSize: 8, nEmbd: 8, nLayer: 1, nHead: 2, seed });
}

export { Rng };
