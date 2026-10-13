import {
  addBias,
  clear,
  copyInto,
  addInto,
  makeLayerNormCache,
  matmulNN,
  matmulNT,
  matmulTN,
  geluBackwardInPlace,
  geluInPlace,
  layernormBackward,
  layernormForward,
  crossEntropy,
  crossEntropyBackward,
  sumRowsInto,
  zeros,
  type LayerNormCache,
  type Mat,
} from '../core/mat.ts';
import { Rng } from '../core/rng.ts';
import { BaseModel, type AttentionMap, type ModelConfig } from './base.ts';
import { makeParam, paramGradMat, paramMat, type Param } from './param.ts';

/**
 * A decoder-only transformer, written out by hand.
 *
 * Pre-norm blocks, multi-head causal self-attention, tied output projection
 * (`logits = x @ wteᵀ`). Forward and backward are explicit loops over flat
 * Float32Arrays: no graph, no allocations per step, which is what makes it
 * fast enough to train a real model inside a browser tab.
 *
 * The gradient of every kernel used here is checked against an independent
 * tape-based autodiff implementation in src/model/gpt.test.ts.
 */

interface GptLayerParams {
  ln1w: Param;
  ln1b: Param;
  attnW: Param;
  attnB: Param;
  attnProjW: Param;
  attnProjB: Param;
  ln2w: Param;
  ln2b: Param;
  fcW: Param;
  fcB: Param;
  projW: Param;
  projB: Param;
}

interface LayerActs {
  xIn: Mat;
  ln1: Mat;
  ln1Cache: LayerNormCache;
  qkv: Mat;
  attnOut: Mat;
  attnProj: Mat;
  xMid: Mat;
  ln2: Mat;
  ln2Cache: LayerNormCache;
  h1pre: Mat;
  h1: Mat;
  h2: Mat;
}

interface ShapeBuffers {
  x: Mat;
  xFinal: Mat;
  lnf: Mat;
  lnfCache: LayerNormCache;
  logits: Mat;
  probs: Mat;
  dlogits: Mat;
  dlnf: Mat;
  dx: Mat;
  dH2: Mat;
  dH1: Mat;
  dH1pre: Mat;
  dLn2: Mat;
  dAttnProj: Mat;
  dAttnOut: Mat;
  dQkv: Mat;
  dLn1: Mat;
  acts: LayerActs[];
  attBuf: Float32Array;
  rowDatt: Float32Array;
  rowDs: Float32Array;
  attnVis: Float32Array[];
  attention: AttentionMap[];
}

/**
 * All the activation storage for one (N, B, T) shape.
 *
 * Training and sampling alternate between a handful of fixed shapes (a big
 * batch of windows, and a single window during generation). Allocating on every
 * switch would churn tens of megabytes per second, so each shape keeps its own
 * buffers and `ensure` simply installs the right set.
 */
function allocateShape(config: ModelConfig, N: number, B: number, T: number): ShapeBuffers {
  const C = config.nEmbd;
  const F = 4 * C;
  const V = config.vocabSize;
  const L = config.nLayer;
  const H = config.nHead;

  const acts: LayerActs[] = [];
  for (let l = 0; l < L; l++) {
    acts.push({
      xIn: zeros(N, C),
      ln1: zeros(N, C),
      ln1Cache: makeLayerNormCache(N, C),
      qkv: zeros(N, 3 * C),
      attnOut: zeros(N, C),
      attnProj: zeros(N, C),
      xMid: zeros(N, C),
      ln2: zeros(N, C),
      ln2Cache: makeLayerNormCache(N, C),
      h1pre: zeros(N, F),
      h1: zeros(N, F),
      h2: zeros(N, C),
    });
  }

  const attnVis: Float32Array[] = [];
  const attention: AttentionMap[] = [];
  for (let l = 0; l < L; l++) {
    for (let h = 0; h < H; h++) {
      const w = new Float32Array(Math.max(1, T * T));
      attnVis.push(w);
      attention.push({ layer: l, head: h, weights: w, t: T });
    }
  }

  return {
    x: zeros(N, C),
    xFinal: zeros(N, C),
    lnf: zeros(N, C),
    lnfCache: makeLayerNormCache(N, C),
    logits: zeros(N, V),
    probs: zeros(N, V),
    dlogits: zeros(N, V),
    dlnf: zeros(N, C),
    dx: zeros(N, C),
    dH2: zeros(N, C),
    dH1: zeros(N, F),
    dH1pre: zeros(N, F),
    dLn2: zeros(N, C),
    dAttnProj: zeros(N, C),
    dAttnOut: zeros(N, C),
    dQkv: zeros(N, 3 * C),
    dLn1: zeros(N, C),
    acts,
    attBuf: new Float32Array(Math.max(1, B * H * T * T)),
    rowDatt: new Float32Array(Math.max(1, T)),
    rowDs: new Float32Array(Math.max(1, T)),
    attnVis,
    attention,
  };
}

export class GptModel extends BaseModel {
  private wte!: Param;
  private wpe!: Param;
  private block: GptLayerParams[] = [];
  private lnfW!: Param;
  private lnfB!: Param;

  // activations
  private acts: LayerActs[] = [];
  private x!: Mat;
  private xFinal!: Mat;
  private lnf!: Mat;
  private lnfCache!: LayerNormCache;

  // gradients
  private dlogits!: Mat;
  private dlnf!: Mat;
  private dx!: Mat;
  private dH2!: Mat;
  private dH1!: Mat;
  private dH1pre!: Mat;
  private dLn2!: Mat;
  private dAttnProj!: Mat;
  private dAttnOut!: Mat;
  private dQkv!: Mat;
  private dLn1!: Mat;

  // attention scratch
  private attBuf: Float32Array = new Float32Array(0);
  private rowDatt: Float32Array = new Float32Array(0);
  private rowDs: Float32Array = new Float32Array(0);
  private attnVis: Float32Array[] = [];

  private bufN = -1;
  private bufB = -1;
  private bufT = -1;

  constructor(config: ModelConfig) {
    super(config);
    const { vocabSize: V, blockSize: T, nEmbd: C, nLayer: L, nHead: H, seed } = config;
    if (C % H !== 0) throw new Error(`nEmbd (${C}) must be divisible by nHead (${H})`);
    const rng = new Rng(seed);
    const std = 0.02;

    this.wte = this.register(makeParam('wte', V, C, () => rng.normal(std)));
    this.wpe = this.register(makeParam('wpe', T, C, () => rng.normal(std)));

    for (let l = 0; l < L; l++) {
      const p: GptLayerParams = {
        ln1w: this.register(makeParam(`h${l}.ln1.weight`, C, 1, () => 1)),
        ln1b: this.register(makeParam(`h${l}.ln1.bias`, C, 1, () => 0)),
        attnW: this.register(makeParam(`h${l}.attn.qkv.weight`, 3 * C, C, () => rng.normal(std))),
        attnB: this.register(makeParam(`h${l}.attn.qkv.bias`, 3 * C, 1, () => 0)),
        attnProjW: this.register(makeParam(`h${l}.attn.proj.weight`, C, C, () => rng.normal(std / Math.sqrt(2 * L)))),
        attnProjB: this.register(makeParam(`h${l}.attn.proj.bias`, C, 1, () => 0)),
        ln2w: this.register(makeParam(`h${l}.ln2.weight`, C, 1, () => 1)),
        ln2b: this.register(makeParam(`h${l}.ln2.bias`, C, 1, () => 0)),
        fcW: this.register(makeParam(`h${l}.mlp.fc.weight`, 4 * C, C, () => rng.normal(std))),
        fcB: this.register(makeParam(`h${l}.mlp.fc.bias`, 4 * C, 1, () => 0)),
        projW: this.register(makeParam(`h${l}.mlp.proj.weight`, C, 4 * C, () => rng.normal(std / Math.sqrt(2 * L)))),
        projB: this.register(makeParam(`h${l}.mlp.proj.bias`, C, 1, () => 0)),
      };
      this.block.push(p);
    }
    this.lnfW = this.register(makeParam('lnf.weight', C, 1, () => 1));
    this.lnfB = this.register(makeParam('lnf.bias', C, 1, () => 0));
  }

  private shapes = new Map<string, ShapeBuffers>();
  private shapeKey = '';

  private ensure(N: number, B: number, T: number): void {
    const key = `${N}|${B}|${T}`;
    if (this.shapeKey === key) return;
    let s = this.shapes.get(key);
    if (!s) {
      // two live shapes at most: the training batch and the generation window
      if (this.shapes.size >= 2) this.shapes.clear();
      s = allocateShape(this.config, N, B, T);
      this.shapes.set(key, s);
    }
    this.install(s, N, B, T);
    this.shapeKey = key;
  }

  private install(s: ShapeBuffers, N: number, B: number, T: number): void {
    this.x = s.x;
    this.xFinal = s.xFinal;
    this.lnf = s.lnf;
    this.lnfCache = s.lnfCache;
    this.logits = s.logits;
    this.probs = s.probs;
    this.dlogits = s.dlogits;
    this.dlnf = s.dlnf;
    this.dx = s.dx;
    this.dH2 = s.dH2;
    this.dH1 = s.dH1;
    this.dH1pre = s.dH1pre;
    this.dLn2 = s.dLn2;
    this.dAttnProj = s.dAttnProj;
    this.dAttnOut = s.dAttnOut;
    this.dQkv = s.dQkv;
    this.dLn1 = s.dLn1;
    this.acts = s.acts;
    this.attBuf = s.attBuf;
    this.rowDatt = s.rowDatt;
    this.rowDs = s.rowDs;
    this.attnVis = s.attnVis;
    this.attention = s.attention;
    this.bufN = N;
    this.bufB = B;
    this.bufT = T;
  }

  forward(tokens: Int32Array, targets: Int32Array, B: number, T: number): number {
    this.cacheBatch(tokens, targets);
    this.ensure(B * T, B, T);
    const C = this.config.nEmbd;
    const V = this.config.vocabSize;
    const L = this.config.nLayer;
    const N = B * T;
    const wte = this.wte.data;
    const wpe = this.wpe.data;
    const x = this.x.data;

    // x = token embedding + position embedding
    for (let n = 0; n < N; n++) {
      const tok = this.tokens[n];
      const t = n % T;
      const srcT = tok * C;
      const srcP = t * C;
      const dst = n * C;
      for (let c = 0; c < C; c++) x[dst + c] = wte[srcT + c] + wpe[srcP + c];
    }

    for (let l = 0; l < L; l++) {
      const p = this.block[l];
      const a = this.acts[l];
      copyInto(a.xIn, this.x);

      layernormForward(this.x, p.ln1w.data, p.ln1b.data, a.ln1, a.ln1Cache);
      matmulNT(a.ln1, paramMat(p.attnW), a.qkv);
      addBias(a.qkv, p.attnB.data);
      this.attnForward(l, B, T);
      matmulNT(a.attnOut, paramMat(p.attnProjW), a.attnProj);
      addBias(a.attnProj, p.attnProjB.data);

      copyInto(a.xMid, this.x);
      addInto(a.xMid, a.attnProj);
      copyInto(this.x, a.xMid);

      layernormForward(this.x, p.ln2w.data, p.ln2b.data, a.ln2, a.ln2Cache);
      matmulNT(a.ln2, paramMat(p.fcW), a.h1pre);
      addBias(a.h1pre, p.fcB.data);
      copyInto(a.h1, a.h1pre);
      geluInPlace(a.h1);
      matmulNT(a.h1, paramMat(p.projW), a.h2);
      addBias(a.h2, p.projB.data);

      addInto(this.x, a.h2);
    }

    copyInto(this.xFinal, this.x);
    layernormForward(this.xFinal, this.lnfW.data, this.lnfB.data, this.lnf, this.lnfCache);
    // tied output projection: logits = lnf @ wteᵀ
    matmulNT(this.lnf, paramMat(this.wte), this.logits);
    if (this.logits.cols !== V) throw new Error('logits shape drift');
    this.lastLoss = crossEntropy(this.logits, this.targets, this.probs);
    return this.lastLoss;
  }

  /** Causal multi-head self-attention over [B, T] with head size C/H. */
  private attnForward(l: number, B: number, T: number): void {
    const C = this.config.nEmbd;
    const H = this.config.nHead;
    const hs = C / H;
    const scale = 1 / Math.sqrt(hs);
    const qkv = this.acts[l].qkv.data;
    const out = this.acts[l].attnOut.data;
    const att = this.attBuf;
    out.fill(0);

    for (let b = 0; b < B; b++) {
      for (let h = 0; h < H; h++) {
        const hoff = h * hs;
        const attBase = (b * H + h) * T * T;
        const vis = this.attnVis[l * H + h];
        const visBase = b === 0 ? 0 : -1;
        for (let t1 = 0; t1 < T; t1++) {
          const qRow = (b * T + t1) * 3 * C + hoff;
          let max = -Infinity;
          for (let t2 = 0; t2 <= t1; t2++) {
            const kRow = (b * T + t2) * 3 * C + C + hoff;
            let s = 0;
            for (let d = 0; d < hs; d++) s += qkv[qRow + d] * qkv[kRow + d];
            s *= scale;
            att[attBase + t1 * T + t2] = s;
            if (s > max) max = s;
          }
          let sum = 0;
          for (let t2 = 0; t2 <= t1; t2++) {
            const e = Math.exp(att[attBase + t1 * T + t2] - max);
            att[attBase + t1 * T + t2] = e;
            sum += e;
          }
          const inv = sum > 0 ? 1 / sum : 0;
          const oRow = (b * T + t1) * C + hoff;
          for (let t2 = 0; t2 <= t1; t2++) {
            const a = att[attBase + t1 * T + t2] * inv;
            att[attBase + t1 * T + t2] = a;
            if (visBase >= 0) vis[t1 * T + t2] = a;
            if (a === 0) continue;
            const vRow = (b * T + t2) * 3 * C + 2 * C + hoff;
            for (let d = 0; d < hs; d++) out[oRow + d] += a * qkv[vRow + d];
          }
        }
      }
    }
  }

  private attnBackward(l: number, B: number, T: number): void {
    const C = this.config.nEmbd;
    const H = this.config.nHead;
    const hs = C / H;
    const scale = 1 / Math.sqrt(hs);
    const qkv = this.acts[l].qkv.data;
    const att = this.attBuf;
    const dOut = this.dAttnOut.data;
    const dQkv = this.dQkv.data;
    dQkv.fill(0);
    const dattRow = this.rowDatt;
    const dsRow = this.rowDs;

    for (let b = 0; b < B; b++) {
      for (let h = 0; h < H; h++) {
        const hoff = h * hs;
        const attBase = (b * H + h) * T * T;
        for (let t1 = 0; t1 < T; t1++) {
          const oRow = (b * T + t1) * C + hoff;
          const qRow = (b * T + t1) * 3 * C + hoff;
          let dot = 0;
          for (let t2 = 0; t2 <= t1; t2++) {
            const vRow = (b * T + t2) * 3 * C + 2 * C + hoff;
            let s = 0;
            for (let d = 0; d < hs; d++) s += dOut[oRow + d] * qkv[vRow + d];
            dattRow[t2] = s;
            const a = att[attBase + t1 * T + t2];
            dot += s * a;
            // dv contribution
            for (let d = 0; d < hs; d++) dQkv[vRow + d] += a * dOut[oRow + d];
          }
          for (let t2 = 0; t2 <= t1; t2++) {
            const a = att[attBase + t1 * T + t2];
            dsRow[t2] = a * (dattRow[t2] - dot);
          }
          for (let t2 = 0; t2 <= t1; t2++) {
            const ds = dsRow[t2];
            if (ds === 0) continue;
            const kRow = (b * T + t2) * 3 * C + C + hoff;
            const k = ds * scale;
            for (let d = 0; d < hs; d++) {
              dQkv[qRow + d] += k * qkv[kRow + d];
              dQkv[kRow + d] += k * qkv[qRow + d];
            }
          }
        }
      }
    }
  }

  backward(): void {
    const C = this.config.nEmbd;
    const T = this.bufT;
    const B = this.bufB;
    const L = this.config.nLayer;

    crossEntropyBackward(this.dlogits, this.probs, this.targets);
    // dlnf = dlogits @ wte ; dwte += dlogitsᵀ @ lnf
    matmulNN(this.dlogits, paramMat(this.wte), this.dlnf);
    matmulTN(this.dlogits, this.lnf, paramGradMat(this.wte), true);

    // final layer norm feeds the residual stream
    clear(this.dx);
    layernormBackward(this.dlnf, this.xFinal, this.lnfW.data, this.lnfCache, this.dx, this.lnfW.grad, this.lnfB.grad);

    for (let l = L - 1; l >= 0; l--) {
      const p = this.block[l];
      const a = this.acts[l];

      // ---- MLP branch: h2 = gelu(ln2 @ fcWᵀ) @ projWᵀ
      copyInto(this.dH2, this.dx);
      matmulNN(this.dH2, paramMat(p.projW), this.dH1);
      matmulTN(this.dH2, a.h1, paramGradMat(p.projW), true);
      sumRowsInto(p.projB.grad, this.dH2);
      copyInto(this.dH1pre, this.dH1);
      geluBackwardInPlace(this.dH1pre, a.h1pre.data);
      matmulNN(this.dH1pre, paramMat(p.fcW), this.dLn2);
      matmulTN(this.dH1pre, a.ln2, paramGradMat(p.fcW), true);
      sumRowsInto(p.fcB.grad, this.dH1pre);
      // layer-norm gradient accumulates into the xMid gradient sitting in dx
      layernormBackward(this.dLn2, a.xMid, p.ln2w.data, a.ln2Cache, this.dx, p.ln2w.grad, p.ln2b.grad);

      // ---- attention branch: attnProj = attnOut @ projWᵀ
      copyInto(this.dAttnProj, this.dx);
      matmulNN(this.dAttnProj, paramMat(p.attnProjW), this.dAttnOut);
      matmulTN(this.dAttnProj, a.attnOut, paramGradMat(p.attnProjW), true);
      sumRowsInto(p.attnProjB.grad, this.dAttnProj);
      this.attnBackward(l, B, T);
      matmulNN(this.dQkv, paramMat(p.attnW), this.dLn1);
      matmulTN(this.dQkv, a.ln1, paramGradMat(p.attnW), true);
      sumRowsInto(p.attnB.grad, this.dQkv);
      // layer-norm gradient accumulates into the layer-input gradient in dx
      layernormBackward(this.dLn1, a.xIn, p.ln1w.data, a.ln1Cache, this.dx, p.ln1w.grad, p.ln1b.grad);
    }

    // embedding gradients: scatter one-hot rows
    const wteGrad = this.wte.grad;
    const wpeGrad = this.wpe.grad;
    const dx = this.dx.data;
    const N = this.bufN;
    for (let n = 0; n < N; n++) {
      const tok = this.tokens[n];
      const t = n % T;
      const srcT = tok * C;
      const srcP = t * C;
      const src = n * C;
      for (let c = 0; c < C; c++) {
        const g = dx[src + c];
        wteGrad[srcT + c] += g;
        wpeGrad[srcP + c] += g;
      }
    }
  }

  override embeddingMatrix(): Mat | null {
    return paramMat(this.wte);
  }

  /** Position embeddings, exposed for the "space" view. */
  positionMatrix(): Mat {
    return paramMat(this.wpe);
  }
}
