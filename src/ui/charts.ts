/**
 * Canvas drawing. No charting library — every pixel here is drawn from the
 * model's own numbers, which is the whole point of the page.
 */
import type { AttentionSnapshot, EmbeddingSnapshot } from '../train/viz.ts';

const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace';

interface Surface {
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;
}

/** Size a canvas for the device pixel ratio and return its 2D context. */
export function surface(canvas: HTMLCanvasElement, cssHeight: number): Surface {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(120, Math.floor(rect.width));
  const h = cssHeight;
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  canvas.style.height = `${h}px`;
  const ctx = canvas.getContext('2d')!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h };
}

function frame(s: Surface, title = ''): void {
  const { ctx, w, h } = s;
  ctx.strokeStyle = '#1e242e';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  if (title) {
    ctx.fillStyle = '#5b6675';
    ctx.font = `10px ${MONO}`;
    ctx.fillText(title, 8, 13);
  }
}

export interface LossPoint {
  step: number;
  loss: number;
  valLoss?: number;
}

/**
 * Loss on a log scale: the interesting part of a language model's training
 * curve is the slow crawl from "random" toward "structure", and a linear axis
 * hides most of it.
 */
export function drawLoss(canvas: HTMLCanvasElement, history: LossPoint[], baseline?: number): void {
  const s = surface(canvas, 260);
  const { ctx, w, h } = s;
  frame(s);
  const padL = 38;
  const padR = 10;
  const padT = 24;
  const padB = 20;
  const plotW = w - padL - padR;
  const plotH = h - padT - padB;
  if (history.length === 0) return;

  const values = history.map((p) => p.loss).concat(history.filter((p) => p.valLoss !== undefined).map((p) => p.valLoss!));
  const hi = Math.max(...values, baseline ?? 0) * 1.05;
  const lo = Math.max(0.05, Math.min(...values) * 0.85);
  const logHi = Math.log(hi);
  const logLo = Math.log(lo);

  const maxStep = Math.max(10, history[history.length - 1].step);
  const x = (step: number): number => padL + (step / maxStep) * plotW;
  const y = (loss: number): number => padT + plotH - ((Math.log(Math.max(loss, lo)) - logLo) / (logHi - logLo)) * plotH;

  // grid lines on a 1-2-5 log ladder
  ctx.font = `9.5px ${MONO}`;
  ctx.textAlign = 'right';
  for (const v of [1, 1.5, 2, 3, 4, 5, 6, 8, 10, 15, 20]) {
    if (v > hi || v < lo) continue;
    const yy = y(v);
    ctx.strokeStyle = '#161b23';
    ctx.beginPath();
    ctx.moveTo(padL, yy);
    ctx.lineTo(w - padR, yy);
    ctx.stroke();
    ctx.fillStyle = '#4c5663';
    ctx.fillText(v.toString(), padL - 6, yy + 3);
  }
  ctx.textAlign = 'left';

  if (baseline !== undefined && baseline <= hi && baseline >= lo) {
    const yy = y(baseline);
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = '#39404b';
    ctx.beginPath();
    ctx.moveTo(padL, yy);
    ctx.lineTo(w - padR, yy);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#5b6675';
    ctx.fillText(`uniform guessing (ln ${baseline.toFixed(2)})`, padL + 6, yy - 4);
  }

  // train loss
  ctx.strokeStyle = '#6ee7a8';
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  for (let i = 0; i < history.length; i++) {
    const p = history[i];
    const px = x(p.step);
    const py = y(p.loss);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // held-out loss markers
  ctx.fillStyle = '#64b5ff';
  for (const p of history) {
    if (p.valLoss === undefined) continue;
    ctx.beginPath();
    ctx.arc(x(p.step), y(p.valLoss), 2.2, 0, Math.PI * 2);
    ctx.fill();
  }

  // last value + step axis
  const last = history[history.length - 1];
  ctx.fillStyle = '#6ee7a8';
  ctx.beginPath();
  ctx.arc(x(last.step), y(last.loss), 3, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#8b97a8';
  ctx.font = `10px ${MONO}`;
  ctx.fillText(`${last.loss.toFixed(3)}`, Math.min(x(last.step) + 6, w - 46), y(last.loss) - 6);
  ctx.fillStyle = '#4c5663';
  ctx.fillText('step 0', padL, h - 6);
  const stepLabel = `step ${maxStep}`;
  ctx.fillText(stepLabel, w - padR - ctx.measureText(stepLabel).width, h - 6);

  ctx.fillStyle = '#6ee7a8';
  ctx.fillText('▬ train', padL + 4, 14);
  ctx.fillStyle = '#64b5ff';
  ctx.fillText('● held-out', padL + 62, 14);
}

/** Top next-token probabilities as horizontal bars. */
export function drawBars(
  canvas: HTMLCanvasElement,
  items: { label: string; value: number; caption?: string }[],
  height: number,
  format: (v: number) => string = (v) => v.toFixed(3),
): void {
  const s = surface(canvas, height);
  const { ctx, w, h } = s;
  frame(s);
  if (items.length === 0) return;
  const padL = 34;
  const padR = 62;
  const padT = 20;
  const rowH = Math.min(24, (h - padT - 8) / items.length);
  const max = Math.max(...items.map((i) => i.value), 1e-6);
  const barW = w - padL - padR;

  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    const yy = padT + i * rowH;
    ctx.fillStyle = i === 0 ? '#e8edf4' : '#8b97a8';
    ctx.font = `11px ${MONO}`;
    ctx.textAlign = 'right';
    ctx.fillText(it.label.length > 8 ? `${it.label.slice(0, 7)}…` : it.label, padL - 6, yy + rowH * 0.66);
    ctx.textAlign = 'left';
    const bw = Math.max(1, (it.value / max) * barW);
    ctx.fillStyle = i === 0 ? 'rgba(110,231,168,0.85)' : 'rgba(100,181,255,0.5)';
    ctx.fillRect(padL, yy + 3, bw, rowH - 7);
    ctx.fillStyle = '#8b97a8';
    ctx.font = `10px ${MONO}`;
    ctx.fillText(format(it.value), padL + bw + 6, yy + rowH * 0.66);
  }
}

/** Attention heatmap for one (layer, head). */
export function drawAttention(canvas: HTMLCanvasElement, map: AttentionSnapshot | null, chars?: string[]): void {
  const s = surface(canvas, 260);
  const { ctx, w, h } = s;
  frame(s);
  if (!map) {
    ctx.fillStyle = '#5b6675';
    ctx.font = `12px ${MONO}`;
    ctx.fillText('attention exists only in the transformer', 12, h / 2);
    return;
  }
  const t = map.t;
  const size = Math.min(w - 26, h - 26);
  const cell = size / t;
  const ox = 20;
  const oy = 14;

  for (let i = 0; i < t; i++) {
    for (let j = 0; j <= i; j++) {
      const v = map.weights[i * t + j];
      // perceptual ramp: near-black → teal → white
      const r = Math.round(12 + 243 * Math.pow(v, 1.6));
      const g = Math.round(20 + 220 * Math.pow(v, 0.85));
      const b = Math.round(32 + 200 * Math.pow(v, 1.2));
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(ox + j * cell, oy + i * cell, Math.ceil(cell), Math.ceil(cell));
    }
  }
  ctx.strokeStyle = '#2a323d';
  ctx.strokeRect(ox + 0.5, oy + 0.5, size, size);

  if (chars && cell >= 7) {
    ctx.font = `${Math.min(9, cell * 0.9)}px ${MONO}`;
    ctx.fillStyle = '#7c8899';
    ctx.textAlign = 'center';
    for (let i = 0; i < t; i++) {
      const ch = chars[i];
      if (ch === undefined) continue;
      ctx.fillText(ch === ' ' ? '␣' : ch === '\n' ? '⏎' : ch, ox + (i + 0.5) * cell, oy + size + 11);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i < t; i++) {
      const ch = chars[i];
      if (ch === undefined) continue;
      ctx.fillText(ch === ' ' ? '␣' : ch === '\n' ? '⏎' : ch, ox - 4, oy + (i + 0.75) * cell);
    }
    ctx.textAlign = 'left';
  }
  ctx.fillStyle = '#5b6675';
  ctx.font = `10px ${MONO}`;
  ctx.fillText(`layer ${map.layer} · head ${map.head} · each row: where that position looks (row i can see 0..i)`, ox, h - 4);
}

/** PCA scatter of the token embeddings, labelled with the characters. */
export function drawEmbedding(canvas: HTMLCanvasElement, snap: EmbeddingSnapshot | null): void {
  const s = surface(canvas, 260);
  const { ctx, w, h } = s;
  frame(s);
  if (!snap) {
    ctx.fillStyle = '#5b6675';
    ctx.font = `12px ${MONO}`;
    ctx.fillText('no embedding matrix for this architecture', 12, h / 2);
    return;
  }
  const pad = 18;
  const pw = w - pad * 2;
  const ph = h - pad * 2;
  ctx.strokeStyle = '#161b23';
  ctx.strokeRect(pad + 0.5, pad + 0.5, pw, ph);

  // hull of the points, to give a sense of spread
  ctx.beginPath();
  snap.points.forEach(([px, py], i) => {
    const x = pad + px * pw;
    const y = pad + (1 - py) * ph;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = 'rgba(110,231,168,0.16)';
  ctx.lineWidth = 1;
  ctx.stroke();

  ctx.font = `10px ${MONO}`;
  snap.points.forEach(([px, py], i) => {
    const x = pad + px * pw;
    const y = pad + (1 - py) * ph;
    const label = snap.labels[i] ?? '';
    const space = label === '␣';
    const newline = label === '⏎';
    ctx.fillStyle = space || newline ? 'rgba(255,179,71,0.75)' : 'rgba(110,231,168,0.9)';
    ctx.beginPath();
    ctx.arc(x, y, space || newline ? 4 : 2.4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = space || newline ? 'rgba(255,179,71,0.95)' : 'rgba(232,237,244,0.82)';
    ctx.fillText(label, x + 5, y + 3.5);
  });
  ctx.fillStyle = '#5b6675';
  ctx.font = `10px ${MONO}`;
  ctx.fillText('two principal components of wte · close together = used in similar contexts', pad, h - 4);
}

/** Per-tensor gradient norms, log scale. */
export function drawGradNorms(canvas: HTMLCanvasElement, items: { name: string; gradNorm: number }[]): void {
  const s = surface(canvas, 200);
  const { ctx, w, h } = s;
  frame(s);
  if (items.length === 0) return;
  const padL = 92;
  const padR = 46;
  const padT = 18;
  const rowH = Math.min(15, (h - padT - 6) / items.length);
  const maxLog = Math.log10(Math.max(...items.map((i) => Math.max(i.gradNorm, 1e-7))));
  const minLog = Math.max(-7, Math.log10(Math.max(Math.min(...items.map((i) => Math.max(i.gradNorm, 1e-7))), 1e-7)));
  const span = Math.max(0.5, maxLog - minLog);
  const barW = w - padL - padR;

  ctx.font = `10px ${MONO}`;
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    const yy = padT + i * rowH;
    const frac = (Math.log10(Math.max(it.gradNorm, 1e-7)) - minLog) / span;
    ctx.fillStyle = '#8b97a8';
    ctx.textAlign = 'right';
    ctx.fillText(it.name.length > 18 ? `${it.name.slice(0, 17)}…` : it.name, padL - 6, yy + rowH * 0.72);
    ctx.textAlign = 'left';
    ctx.fillStyle = 'rgba(100,181,255,0.55)';
    ctx.fillRect(padL, yy + 2, Math.max(1, frac * barW), Math.max(2, rowH - 5));
    ctx.fillStyle = '#5b6675';
    ctx.fillText(it.gradNorm.toExponential(1), padL + barW + 6, yy + rowH * 0.72);
  }
  ctx.fillStyle = '#5b6675';
  ctx.fillText('log scale · a dead head shows up here as a norm several orders below its neighbours', padL, h - 4);
}
