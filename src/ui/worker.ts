/// <reference lib="webworker" />
/**
 * The training worker.
 *
 * The whole model lives here, on a thread of its own, so the page keeps
 * painting at 60fps while it chews through batches. It drip-feeds three kinds
 * of data to the UI:
 *
 *   - metrics  every tick: loss, learning rate, gradient norm, tokens/second
 *   - visuals  throttled: attention maps, PCA embedding, per-tensor grad norms
 *   - text     a few characters per tick, from a streaming `Generator`
 *
 * One subtlety worth naming: every forward pass the worker performs — training
 * batches, the prompt inspection, the sample generator — uses one of two fixed
 * tensor shapes, so `GptModel.ensure` never has to reallocate mid-run.
 */
import { Tokenizer } from '../data/tokenizer.ts';
import { configFor, createModel } from '../model/factory.ts';
import type { BaseModel, ModelKind } from '../model/base.ts';
import { Trainer } from '../train/trainer.ts';
import { captureEmbedding, type AttentionSnapshot } from '../train/viz.ts';
import { decodeCheckpoint, encodeCheckpointString } from '../train/checkpoint.ts';
import { estimateFlops, formatFlops } from '../model/flops.ts';
import { describeGraph, type GraphRow } from '../model/graph.ts';
import { Generator } from '../model/sample.ts';
import { PRESETS } from './config.ts';
import type { GradRow, MainToWorker, Metrics, NextRow, WorkerToMain } from './protocol.ts';

const post = (msg: WorkerToMain): void => (self as unknown as Worker).postMessage(msg);

const SAMPLE_LENGTH = 220;
const SAMPLE_TOKENS_PER_TICK = 2;
const TICK_BUDGET_MS = 45;
const NEXT_MS = 320;
const GRAD_MS = 900;
const SPACE_MS = 2200;

interface Session {
  model: BaseModel;
  tokenizer: Tokenizer;
  trainer: Trainer;
  arch: ModelKind;
  corpusName: string;
  corpusChars: number;
  warmStarted: boolean;
  running: boolean;
  finished: boolean;
  stepMs: number;
  smoothLoss: number;
  elapsedMs: number;
  prompt: string;
  promptTokens: Int32Array;
  promptTargets: Int32Array;
  selected: { layer: number; head: number };
  generator: Generator;
  sampleText: string;
  attnSnap: AttentionSnapshot | null;
  lastLoss: number;
  lastGradNorm: number;
  nodes: GraphRow[];
  lastNext: number;
  lastGrad: number;
  lastSpace: number;
  heads: { layer: number; head: number }[];
}

let S: Session | null = null;

const now = (): number => performance.now();

function label(ch: string | undefined): string {
  if (ch === undefined) return '?';
  if (ch === ' ') return '␣';
  if (ch === '\n') return '⏎';
  return ch;
}

function metrics(s: Session): Metrics {
  const flops = estimateFlops(s.model.config, s.trainer.cfg.batchSize, s.trainer.cfg.blockSize);
  return {
    step: s.trainer.step,
    plannedSteps: s.trainer.cfg.steps,
    loss: s.lastLoss,
    smoothLoss: s.smoothLoss,
    valLoss: s.trainer.lastValLoss,
    lr: s.trainer.opt.lrAt(s.trainer.step),
    gradNorm: s.lastGradNorm,
    tokensPerSec: (s.trainer.cfg.batchSize * s.trainer.cfg.blockSize) / (Math.max(s.stepMs, 0.01) / 1000),
    stepMs: s.stepMs,
    elapsedMs: s.elapsedMs,
    paramCount: s.model.numParams(),
    arch: s.arch,
    config: s.model.config,
    vocabSize: s.tokenizer.vocabSize,
    context: s.model.config.blockSize,
    flopsPerStep: flops.flops,
    flopsLabel: formatFlops(flops.flops),
    running: s.running,
    finished: s.finished,
    warmStarted: s.warmStarted,
    corpusName: s.corpusName,
    corpusChars: s.corpusChars,
  };
}

/** Top-8 next-character distribution after `prompt`, using the training shape. */
function nextDistribution(s: Session): NextRow[] {
  const B = s.trainer.cfg.batchSize;
  const T = s.trainer.cfg.blockSize;
  const ids = Array.from(s.tokenizer.encode(s.prompt)).slice(-T);
  const tokens = s.promptTokens;
  tokens.fill(0);
  for (let i = 0; i < ids.length; i++) tokens[i] = ids[i];
  const row = Math.max(0, ids.length - 1);
  s.model.forward(tokens, s.promptTargets, B, T);
  const V = s.model.config.vocabSize;
  const off = row * V;
  const logits = s.model.logits.data;

  let max = -Infinity;
  for (let i = 0; i < V; i++) if (logits[off + i] > max) max = logits[off + i];
  let sum = 0;
  const probs = new Float64Array(V);
  for (let i = 0; i < V; i++) {
    const p = Math.exp(logits[off + i] - max);
    probs[i] = p;
    sum += p;
  }
  const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
  return order.slice(0, 8).map((i) => ({ char: label(s.tokenizer.chars[i]), prob: probs[i] / sum }));
}

/** Gradient and weight norms per tensor — the "is anything dead in here?" panel. */
function gradRows(s: Session): GradRow[] {
  const out: GradRow[] = [];
  for (const p of s.model.params) {
    let acc = 0;
    const n = p.data.length;
    for (let i = 0; i < n; i++) acc += p.grad[i] * p.grad[i];
    out.push({ name: p.name, gradNorm: Math.sqrt(acc / Math.max(1, n)) });
  }
  return out;
}

function startSession(msg: Extract<MainToWorker, { type: 'start' }>): Session {
  const preset = PRESETS[msg.preset];
  const overrides = {
    blockSize: preset.blockSize,
    nEmbd: preset.nEmbd,
    nLayer: preset.nLayer,
    nHead: preset.nHead,
    seed: 1337,
  };
  const freshTokenizer = Tokenizer.fromText(msg.corpusText);

  let model: BaseModel | null = null;
  let tokenizer = freshTokenizer;
  let warmStarted = false;

  if (msg.warmStart) {
    try {
      const ckpt = decodeCheckpoint(msg.warmStart);
      const c = ckpt.model.config;
      const sameShape =
        c.kind === msg.arch &&
        c.nEmbd === overrides.nEmbd &&
        c.nLayer === overrides.nLayer &&
        c.nHead === overrides.nHead &&
        c.blockSize === overrides.blockSize &&
        ckpt.tokenizer.chars.join('') === freshTokenizer.chars.join('');
      if (sameShape) {
        model = ckpt.model;
        tokenizer = ckpt.tokenizer;
        warmStarted = true;
      }
    } catch {
      warmStarted = false;
    }
  }
  if (!model) model = createModel(configFor(msg.arch, freshTokenizer.vocabSize, overrides));

  const data = tokenizer.encode(msg.corpusText);
  const trainer = new Trainer(model, tokenizer, data, {
    steps: preset.steps,
    batchSize: preset.batchSize,
    blockSize: preset.blockSize,
    evalEvery: 250,
    evalBatches: 4,
    sampleEvery: Number.MAX_SAFE_INTEGER,
    sampleLength: 1,
    temperature: 0.8,
    topK: 40,
    seed: 1337,
    prompt: msg.prompt,
    optimizer: {
      lr: msg.lr,
      minLr: msg.lr / 12,
      warmupSteps: 60,
      gradClip: 1,
      weightDecay: 0.01,
    },
  });

  const B = preset.batchSize;
  const T = preset.blockSize;
  const heads: { layer: number; head: number }[] = [];
  for (let l = 0; l < model.config.nLayer; l++) {
    for (let h = 0; h < (msg.arch === 'gpt' ? model.config.nHead : 0); h++) heads.push({ layer: l, head: h });
  }

  const promptIds = Array.from(tokenizer.encode(msg.prompt));
  const session: Session = {
    model,
    tokenizer,
    trainer,
    arch: msg.arch,
    corpusName: msg.corpusName,
    corpusChars: msg.corpusText.length,
    warmStarted,
    running: true,
    finished: false,
    stepMs: 4,
    smoothLoss: Number.NaN,
    elapsedMs: 0,
    prompt: msg.prompt,
    promptTokens: new Int32Array(B * T),
    promptTargets: new Int32Array(B * T),
    selected: { layer: Math.min(1, Math.max(0, model.config.nLayer - 1)), head: 0 },
    generator: new Generator(model, promptIds.length > 0 ? promptIds : [0], {
      maxNew: SAMPLE_LENGTH,
      temperature: 0.8,
      topK: 40,
      seed: 7,
    }),
    sampleText: '',
    attnSnap: null,
    lastLoss: Number.NaN,
    lastGradNorm: 0,
    nodes: describeGraph(model.config, B, T),
    lastNext: 0,
    lastGrad: 0,
    lastSpace: 0,
    heads,
  };
  return session;
}

function restartSample(s: Session): void {
  const ids = Array.from(s.tokenizer.encode(s.prompt));
  s.generator = new Generator(s.model, ids.length > 0 ? ids : [0], {
    maxNew: SAMPLE_LENGTH,
    temperature: 0.8,
    topK: 40,
    seed: 7 + s.trainer.step,
  });
  s.sampleText = '';
}

function tick(): void {
  const s = S;
  if (!s || !s.running) return;
  const t0 = now();
  let measured = 0;
  let count = 0;

  while (now() - t0 < TICK_BUDGET_MS && s.trainer.step < s.trainer.cfg.steps) {
    const r = s.trainer.stepOnce();
    measured += r.elapsedMs;
    count++;
    s.lastLoss = r.loss;
    s.lastGradNorm = r.gradNorm;
    s.smoothLoss = Number.isFinite(s.smoothLoss) ? 0.94 * s.smoothLoss + 0.06 * r.loss : r.loss;
    if (s.trainer.step % s.trainer.cfg.evalEvery === 0) s.trainer.evaluate();
  }
  if (count > 0) {
    s.stepMs = measured / count;
    s.elapsedMs += now() - t0;
  }
  if (!s.finished && s.trainer.step >= s.trainer.cfg.steps) {
    s.finished = true;
    s.trainer.evaluate();
  }

  // attention is read straight out of the training forward, before the
  // generator installs its own (smaller) buffer set
  const maps = s.model.attention;
  const wanted = maps.find((m) => m.layer === s.selected.layer && m.head === s.selected.head) ?? maps[0] ?? null;
  s.attnSnap = wanted
    ? {
        layer: wanted.layer,
        head: wanted.head,
        t: wanted.t,
        weights: Array.from(wanted.weights, (v) => Math.round(v * 1000) / 1000),
      }
    : null;

  // the sample streams a couple of characters per tick, so text keeps
  // appearing without stalling training
  const ids = s.generator.advance(SAMPLE_TOKENS_PER_TICK);
  if (ids.length > 0) s.sampleText += s.tokenizer.decode(ids);
  if (s.generator.produced >= s.generator.maxNew) restartSample(s);

  // characters behind the attention map, for the panel's axis labels
  const batch = s.model.lastBatchTokens();
  const chars: string[] = [];
  for (let i = 0; i < Math.min(wanted ? wanted.t : 0, batch.length); i++) {
    const ch = s.tokenizer.chars[batch[i]];
    chars.push(ch === undefined ? '?' : ch === ' ' ? '␣' : ch === '\n' ? '⏎' : ch);
  }

  const t = now();
  const payload: WorkerToMain = {
    type: 'update',
    metrics: metrics(s),
    attention: s.attnSnap,
    embedding: null,
    next: [],
    grads: [],
    chars,
    sampleChunk: s.sampleText,
  };

  if (t - s.lastNext > NEXT_MS) {
    s.lastNext = t;
    payload.next = nextDistribution(s);
  }
  if (t - s.lastGrad > GRAD_MS) {
    s.lastGrad = t;
    payload.grads = gradRows(s);
  }
  if (t - s.lastSpace > SPACE_MS) {
    s.lastSpace = t;
    payload.embedding = captureEmbedding(s.model, s.tokenizer);
  }

  post(payload);
  setTimeout(tick, 0);
}

self.onmessage = (event: MessageEvent<MainToWorker>): void => {
  const msg = event.data;
  try {
    if (msg.type === 'start') {
      S = startSession(msg);
      post({ type: 'started', metrics: metrics(S), graph: S.nodes, heads: S.heads });
      tick();
      return;
    }
    if (!S) return;
    if (msg.type === 'pause') {
      S.running = false;
      post({ type: 'update', metrics: metrics(S), attention: S.attnSnap, embedding: null, sampleChunk: S.sampleText });
      return;
    }
    if (msg.type === 'resume') {
      if (!S.running) {
        S.running = true;
        tick();
      }
      return;
    }
    if (msg.type === 'inspect') {
      S.prompt = msg.prompt;
      restartSample(S);
      S.lastNext = 0;
      return;
    }
    if (msg.type === 'attention') {
      S.selected = { layer: msg.layer, head: msg.head };
      return;
    }
    if (msg.type === 'export') {
      post({
        type: 'exported',
        json: encodeCheckpointString(S.model, S.tokenizer, {
          name: `${S.arch}-${S.corpusName}`,
          corpus: S.corpusName,
          steps: S.trainer.step,
          loss: S.lastLoss,
          valLoss: S.trainer.lastValLoss,
          notes: `${msg.meta} — trained in the browser at emirmizrakli.github.io/emergence`,
        }),
        filename: `emergence-${S.arch}-step${S.trainer.step}.json`,
      });
      return;
    }
  } catch (err) {
    post({ type: 'error', message: err instanceof Error ? err.message : String(err) });
  }
};
