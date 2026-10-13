/**
 * Finite-difference gradient checking.
 *
 * The whole point of this file: every hand-written backward pass in this
 * project is verified against a numerical derivative of its own forward pass.
 * A gradient bug in a hand-derived kernel is otherwise nearly invisible — the
 * model still trains, just worse — which is exactly the kind of failure this
 * project cannot afford to ship.
 *
 * Central differences with a fairly large step: the maths runs in float32, so
 * a tiny step drowns in rounding noise. eps = 2e-2 keeps truncation error and
 * rounding error balanced and reliably reproduces gradients to ~1e-3.
 */
export function numericalGrad(f: (x: Float32Array) => number, x: Float32Array, eps = 2e-2): Float32Array {
  const out = new Float32Array(x.length);
  const probe = x.slice();
  for (let i = 0; i < x.length; i++) {
    const orig = probe[i];
    probe[i] = orig + eps;
    const plus = f(probe);
    probe[i] = orig - eps;
    const minus = f(probe);
    probe[i] = orig;
    out[i] = (plus - minus) / (2 * eps);
  }
  return out;
}

export interface GradComparison {
  ok: boolean;
  maxAbs: number;
  maxRel: number;
  worstIndex: number;
  analytic: number;
  numeric: number;
}

/**
 * Compare an analytic gradient against a numerical one.
 * Entries whose numerical gradient is negligible are judged on absolute error
 * only (relative error is meaningless next to float32 noise).
 */
export function compareGrad(
  analytic: ArrayLike<number>,
  numeric: ArrayLike<number>,
  tolAbs = 2e-3,
  tolRel = 0.05,
  negligible = 1e-3,
): GradComparison {
  let maxAbs = 0;
  let maxRel = 0;
  let worstIndex = -1;
  let worstA = Number.NaN;
  let worstN = Number.NaN;
  let ok = true;
  for (let i = 0; i < analytic.length; i++) {
    const a = analytic[i];
    const n = numeric[i];
    const abs = Math.abs(a - n);
    if (abs > maxAbs) {
      maxAbs = abs;
      worstIndex = i;
      worstA = a;
      worstN = n;
    }
    const scale = Math.max(Math.abs(a), Math.abs(n));
    if (scale > negligible) {
      const rel = abs / scale;
      if (rel > maxRel) maxRel = rel;
      if (abs > tolAbs && rel > tolRel) ok = false;
    } else if (abs > tolAbs) {
      ok = false;
    }
  }
  return { ok, maxAbs, maxRel, worstIndex, analytic: worstA, numeric: worstN };
}

export function describeGrad(c: GradComparison): string {
  return `ok=${c.ok} maxAbs=${c.maxAbs.toExponential(2)} maxRel=${c.maxRel.toExponential(2)} worst@${c.worstIndex} analytic=${c.analytic.toExponential(3)} numeric=${c.numeric.toExponential(3)}`;
}
