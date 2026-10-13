/**
 * Deterministic pseudo-random number generation.
 *
 * Every run in Emergence is reproducible from a seed: initial weights, batch
 * sampling and text generation all draw from instances created here. The
 * generator is mulberry32 — small, fast, and good enough for weight init.
 */

export class Rng {
  private state: number;
  private spare: number | null = null;

  constructor(seed = 1337) {
    // 0 is a fixed point for mulberry32; nudge it.
    this.state = (seed >>> 0) === 0 ? 0x9e3779b9 : seed >>> 0;
  }

  /** Uniform in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform in [lo, hi). */
  range(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }

  /** Integer in [0, n). */
  int(n: number): number {
    return Math.floor(this.next() * n);
  }

  /** Standard normal via Box–Muller, caching the second sample. */
  gauss(): number {
    if (this.spare !== null) {
      const s = this.spare;
      this.spare = null;
      return s;
    }
    let u = 0;
    let v = 0;
    let r = 0;
    do {
      u = this.next() * 2 - 1;
      v = this.next() * 2 - 1;
      r = u * u + v * v;
    } while (r === 0 || r >= 1);
    const f = Math.sqrt((-2 * Math.log(r)) / r);
    this.spare = v * f;
    return u * f;
  }

  /** Normal with the given standard deviation. */
  normal(std: number): number {
    return this.gauss() * std;
  }
}
