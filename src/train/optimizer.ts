import type { Param } from '../model/param.ts';

export interface OptimizerConfig {
  /** Peak learning rate. */
  lr: number;
  /** Floor of the cosine schedule. */
  minLr: number;
  /** Linear warmup length, in steps. */
  warmupSteps: number;
  beta1: number;
  beta2: number;
  eps: number;
  /** Decoupled decay applied to weights (not to biases / norms). */
  weightDecay: number;
  /** Global gradient-norm clip; 0 disables clipping. */
  gradClip: number;
}

export const DEFAULT_OPTIMIZER: OptimizerConfig = {
  lr: 3e-3,
  minLr: 3e-4,
  warmupSteps: 50,
  beta1: 0.9,
  beta2: 0.99,
  eps: 1e-8,
  weightDecay: 0.01,
  gradClip: 1,
};

/**
 * Adam with linear warmup and cosine decay, plus global gradient-norm
 * clipping. Written out rather than imported so the whole training loop stays
 * inside the "no dependencies" claim — and so the learning-rate schedule is
 * something you can read in twenty lines.
 */
export class Adam {
  private params: Param[];
  private cfg: OptimizerConfig;
  /** Total steps budget, used by the cosine schedule. */
  totalSteps: number;
  step0 = 0;
  private clipCoefficient = 1;

  constructor(params: Param[], cfg: Partial<OptimizerConfig> = {}, totalSteps = 1000) {
    this.params = params;
    this.cfg = { ...DEFAULT_OPTIMIZER, ...cfg };
    this.totalSteps = Math.max(1, totalSteps);
  }

  get config(): OptimizerConfig {
    return this.cfg;
  }

  /** Learning rate at a given step: warmup then cosine decay to minLr. */
  lrAt(step: number): number {
    const { lr, minLr, warmupSteps } = this.cfg;
    if (warmupSteps > 0 && step < warmupSteps) return (lr * (step + 1)) / warmupSteps;
    const progress = Math.min(1, (step - warmupSteps) / Math.max(1, this.totalSteps - warmupSteps));
    return minLr + 0.5 * (lr - minLr) * (1 + Math.cos(Math.PI * progress));
  }

  /** Global L2 norm of all gradients, before clipping. */
  gradNorm(): number {
    let sum = 0;
    for (const p of this.params) {
      const g = p.grad;
      for (let i = 0; i < g.length; i++) sum += g[i] * g[i];
    }
    return Math.sqrt(sum);
  }

  /**
   * Apply one Adam update. Returns the learning rate actually used, so the
   * trainer can put it on the chart.
   */
  step(): number {
    const { beta1, beta2, eps, weightDecay, gradClip } = this.cfg;
    const lr = this.lrAt(this.step0);
    const t = this.step0 + 1;

    // gradient clipping, computed once and shared by every parameter
    let clip = 1;
    if (gradClip > 0) {
      const norm = this.gradNorm();
      clip = norm > gradClip ? gradClip / (norm + 1e-6) : 1;
    }
    this.clipCoefficient = clip;

    const bias1 = 1 - Math.pow(beta1, t);
    const bias2 = 1 - Math.pow(beta2, t);

    for (const p of this.params) {
      const { data, grad, m, v } = p;
      const decay = p.name.includes('weight') && p.cols > 1 ? weightDecay : 0;
      for (let i = 0; i < data.length; i++) {
        const g = grad[i] * clip + decay * data[i];
        m[i] = beta1 * m[i] + (1 - beta1) * g;
        v[i] = beta2 * v[i] + (1 - beta2) * g * g;
        const mHat = m[i] / bias1;
        const vHat = v[i] / bias2;
        data[i] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    }
    this.step0++;
    return lr;
  }

  /** The scale factor applied to gradients by the most recent clip. */
  get lastClip(): number {
    return this.clipCoefficient;
  }
}
