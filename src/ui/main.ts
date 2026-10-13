/**
 * Page orchestration: one worker, five canvases, no framework.
 *
 * The worker owns the model; this file only reads the messages it sends and
 * paints them. Every panel is redrawn from live values on an animation frame
 * budget, so nothing on screen is a static mock-up: if the model is untrained,
 * the panels look untrained.
 */
import { drawAttention, drawBars, drawEmbedding, drawGradNorms, drawLoss, type LossPoint } from './charts.ts';
import { CORPORA, DEFAULT_PROMPT, PRESETS, WARM_START_URL } from './config.ts';
import type { GradRow, MainToWorker, Metrics, NextRow, WorkerToMain } from './protocol.ts';
import type { AttentionSnapshot, EmbeddingSnapshot } from '../train/viz.ts';
import type { GraphRow } from '../model/graph.ts';

const $ = <T extends HTMLElement = HTMLElement>(id: string): T => document.getElementById(id) as T;

interface View {
  metrics: Metrics | null;
  history: LossPoint[];
  sample: string;
  next: NextRow[];
  grads: GradRow[];
  attention: AttentionSnapshot | null;
  embedding: EmbeddingSnapshot | null;
  graph: GraphRow[];
  heads: { layer: number; head: number }[];
  selected: { layer: number; head: number };
  chars: string[];
  error: string | null;
}

const view: View = {
  metrics: null,
  history: [],
  sample: '',
  next: [],
  grads: [],
  attention: null,
  embedding: null,
  graph: [],
  heads: [],
  selected: { layer: 1, head: 0 },
  chars: [],
  error: null,
};

let worker: Worker | null = null;
let corpusText = '';
let warmCheckpoint: string | null = null;
let dirty = true;

const num = (n: number): string => n.toLocaleString('en-US');

/* ---------------------------------------------------------------- worker ---- */

function startRun(resetHistory = true): void {
  if (resetHistory) view.history = [];
  if (worker) worker.terminate();

  const arch = ($<HTMLSelectElement>('arch').value || 'gpt') as Metrics['arch'];
  const preset = ($<HTMLSelectElement>('preset').value || 'small') as keyof typeof PRESETS;
  const corpusKey = $<HTMLSelectElement>('corpus').value;
  const useWarm = $<HTMLSelectElement>('warm').value === 'shakespeare';
  const pasted = ($<HTMLTextAreaElement>('corpus-text').value || '').trim();

  const text = corpusKey === 'paste' && pasted.length > 40 ? pasted : corpusText;
  const corpusName = corpusKey === 'paste' && pasted.length > 40 ? 'pasted text' : 'tiny shakespeare';
  const lr = Math.pow(10, Number($<HTMLInputElement>('lr').value));

  $('sample').textContent = 'warming up the first sample…';

  worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
  worker.onmessage = (event: MessageEvent<WorkerToMain>): void => {
    handle(event.data);
  };
  worker.onerror = (): void => {
    view.error = 'worker failed to start — see the browser console';
    dirty = true;
  };

  const msg: MainToWorker = {
    type: 'start',
    arch,
    preset,
    lr,
    corpusText: text,
    corpusName,
    warmStart: useWarm ? warmCheckpoint : null,
    prompt: $<HTMLInputElement>('prompt').value || DEFAULT_PROMPT,
  };
  worker.postMessage(msg);
}

function handle(msg: WorkerToMain): void {
  switch (msg.type) {
    case 'started': {
      view.metrics = msg.metrics;
      view.graph = msg.graph;
      view.heads = msg.heads;
      view.chars = [];
      view.error = null;
      view.selected = { layer: Math.min(1, msg.metrics.config.nLayer - 1), head: 0 };
      buildHeadPicker();
      buildTape();
      $('run').textContent = 'pause';
      dirty = true;
      break;
    }
    case 'update': {
      const m = msg.metrics;
      view.metrics = m;
      if (Number.isFinite(m.smoothLoss) && m.step > 0) {
        const last = view.history[view.history.length - 1];
        const point: LossPoint = { step: m.step, loss: m.smoothLoss };
        if (m.valLoss !== null) point.valLoss = m.valLoss;
        if (!last || last.step !== m.step) view.history.push(point);
        else view.history[view.history.length - 1] = point;
        if (view.history.length > 1400) view.history.splice(0, view.history.length - 1400);
      }
      if (msg.attention) {
        view.attention = msg.attention;
        view.selected = { layer: msg.attention.layer, head: msg.attention.head };
      }
      if (msg.embedding) view.embedding = msg.embedding;
      if (msg.grads && msg.grads.length > 0) view.grads = msg.grads;
      if (msg.next && msg.next.length > 0) view.next = msg.next;
      if (msg.chars && msg.chars.length > 0) view.chars = msg.chars;
      if (msg.sampleChunk !== undefined) view.sample = msg.sampleChunk;
      $('run').textContent = m.running ? 'pause' : 'train';
      dirty = true;
      break;
    }
    case 'exported': {
      const blob = new Blob([msg.json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = $<HTMLAnchorElement>('dl');
      a.href = url;
      a.download = msg.filename;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 4000);
      break;
    }
    case 'error': {
      view.error = msg.message;
      dirty = true;
      break;
    }
  }
}

/* ------------------------------------------------------------------ chrome --- */

function buildHeadPicker(): void {
  const host = $('attn-selector');
  if (view.heads.length === 0) {
    host.textContent = '—';
    return;
  }
  if (host.querySelector('select')) return;
  const select = document.createElement('select');
  select.id = 'attn-pick';
  for (const h of view.heads) {
    const option = document.createElement('option');
    option.value = `${h.layer}:${h.head}`;
    option.textContent = `L${h.layer}H${h.head}`;
    select.appendChild(option);
  }
  select.value = `${view.selected.layer}:${view.selected.head}`;
  select.onchange = (): void => {
    const [layer, head] = select.value.split(':').map(Number);
    view.selected = { layer, head };
    worker?.postMessage({ type: 'attention', layer, head } satisfies MainToWorker);
    dirty = true;
  };
  host.textContent = '';
  host.appendChild(select);
}

function buildTape(): void {
  const table = $<HTMLTableElement>('tape');
  const max = Math.max(...view.graph.map((r) => r.macs), 1);
  const rows = ['<tr><th>operation</th><th>shape</th><th>macs</th><th>cost</th></tr>'];
  for (const r of view.graph) {
    const pct = (r.macs / max) * 100;
    rows.push(
      `<tr><td class="op">${r.op}</td><td>${r.shape}</td><td class="num">${num(r.macs)}</td>` +
        `<td><div class="costbar" style="width:${pct.toFixed(1)}%"></div></td></tr>`,
    );
  }
  table.innerHTML = rows.join('');
}

const secs = (ms: number): string => (ms < 1000 ? `${Math.round(ms)}ms` : ms < 60000 ? `${(ms / 1000).toFixed(1)}s` : `${(ms / 60000).toFixed(1)}min`);

function drawAll(): void {
  const m = view.metrics;

  // status dot
  const dot = $('status-dot');
  dot.className = view.error ? 'dot error' : !m ? 'dot paused' : m.finished ? 'dot done' : m.running ? 'dot running' : 'dot paused';

  $('s-step').textContent = m ? num(m.step) : '—';
  $('s-loss').textContent = m && Number.isFinite(m.smoothLoss) ? m.smoothLoss.toFixed(3) : '—';
  $('s-val').textContent = m && m.valLoss !== null ? m.valLoss.toFixed(3) : '—';
  $('s-lr').textContent = m ? m.lr.toExponential(1) : '—';
  $('s-tps').textContent = m && Number.isFinite(m.tokensPerSec) ? num(Math.round(m.tokensPerSec)) : '—';
  $('s-params').textContent = m ? num(m.paramCount) : '—';
  $('s-flops').textContent = m ? m.flopsLabel : '—';

  const baseline = m ? Math.log(m.vocabSize) : undefined;
  drawLoss($<HTMLCanvasElement>('c-loss'), view.history, baseline);
  drawBars(
    $<HTMLCanvasElement>('c-next'),
    view.next.map((n) => ({ label: n.char, value: n.prob })),
    260,
    (v) => `${(v * 100).toFixed(1)}%`,
  );
  drawAttention($<HTMLCanvasElement>('c-attn'), view.attention, view.chars);
  drawEmbedding($<HTMLCanvasElement>('c-space'), view.embedding);
  drawGradNorms($<HTMLCanvasElement>('c-grad'), view.grads);

  if (view.sample) $('sample').textContent = view.sample;

  if (m) {
    $('h-loss').textContent =
      `${num(m.step)}/${num(m.plannedSteps)} steps · ${secs(m.stepMs)}/step · ${secs(m.elapsedMs)} of compute · ` +
      `learned on ${num(m.corpusChars)} characters · ${num(m.tokensPerSec * (m.elapsedMs / 1000))} characters processed so far`;
    $('h-next').textContent =
      `after "${$<HTMLInputElement>('prompt').value}" — the entire vocabulary is the ${num(m.vocabSize)} distinct characters in the corpus`;
    $('h-attn').textContent = view.attention
      ? `row i = where position i looked while predicting character i+1 · ${view.heads.length} heads total`
      : 'attention exists only in the transformer — switch architecture to see it';
    $('h-space').textContent =
      `learned embeddings of all ${num(m.vocabSize)} characters, projected to 2d (power iteration PCA) · ${m.config.nEmbd}-dimensional vectors · orange = whitespace`;
    $('h-sample').textContent =
      `sampled from the live weights at step ${num(m.step)} · temperature 0.8, top-k 40 · the text above is regenerated as training continues`;
    $('h-tape').textContent = view.graph.length
      ? `forward pass at context ${m.config.blockSize} · ${m.flopsLabel} FLOPs per step (forward + backward) · ${num(
          m.flopsPerStep / (m.stepMs / 1000),
        )} FLOPs/s`
      : '';
    $('h-grad').textContent = view.grads.length
      ? `per-tensor gradient RMS, log scale · global grad norm is clipped to 1.0 · highest: ${view.grads
          .slice()
          .sort((a, b) => b.gradNorm - a.gradNorm)[0]
          ?.name ?? ''}`
      : '';
  }
  if (view.error) $('h-sample').textContent = `⚠ ${view.error}`;
}

function paint(): void {
  if (!dirty) return;
  dirty = false;
  drawAll();
}

function loop(): void {
  paint();
  requestAnimationFrame(loop);
}

// requestAnimationFrame stops while a tab is hidden — a background tab, a phone
// with the screen off, some embedded viewers. Relying on it alone leaves the
// page looking frozen until it is focused again, so paint on a slow timer too,
// and once more the moment the tab comes back.
setInterval(paint, 250);
document.addEventListener('visibilitychange', paint);


/* ---------------------------------------------------------------- controls --- */

function wireControls(): void {
  $('run').addEventListener('click', () => {
    const m = view.metrics;
    if (!m) return;
    if (m.running) worker?.postMessage({ type: 'pause' } satisfies MainToWorker);
    else worker?.postMessage({ type: 'resume' } satisfies MainToWorker);
  });

  $('reset').addEventListener('click', () => {
    startRun(true);
  });

  for (const id of ['arch', 'preset', 'warm']) {
    $(id).addEventListener('change', () => startRun(true));
  }

  $('corpus').addEventListener('change', () => {
    const pasted = $<HTMLSelectElement>('corpus').value === 'paste';
    $<HTMLTextAreaElement>('corpus-text').hidden = !pasted;
    if (!pasted) startRun(true);
    else if (($<HTMLTextAreaElement>('corpus-text').value || '').trim().length > 40) startRun(true);
    dirty = true;
  });

  $<HTMLTextAreaElement>('corpus-text').addEventListener('change', () => {
    if ($<HTMLSelectElement>('corpus').value === 'paste') startRun(true);
  });

  const lr = $<HTMLInputElement>('lr');
  const applyLr = (): void => {
    $('lr-out').textContent = Math.pow(10, Number(lr.value)).toExponential(1);
  };
  applyLr();
  lr.addEventListener('input', applyLr);
  lr.addEventListener('change', () => startRun(true));

  const prompt = $<HTMLInputElement>('prompt');
  const inspect = (): void => {
    worker?.postMessage({ type: 'inspect', prompt: prompt.value } satisfies MainToWorker);
  };
  $('prompt-go').addEventListener('click', inspect);
  prompt.addEventListener('keydown', (e) => {
    if ((e as KeyboardEvent).key === 'Enter') inspect();
  });

  $('dl').addEventListener('click', (e) => {
    e.preventDefault();
    worker?.postMessage({ type: 'export', meta: 'emergence checkpoint' } satisfies MainToWorker);
  });
}

/* ------------------------------------------------------------------- boot ---- */

async function boot(): Promise<void> {
  wireControls();
  requestAnimationFrame(loop);
  $<HTMLInputElement>('prompt').value = DEFAULT_PROMPT;

  try {
    const res = await fetch(CORPORA.shakespeare.url);
    if (!res.ok) throw new Error(`${res.status}`);
    corpusText = await res.text();
  } catch {
    corpusText = `the corpus could not be loaded.\n`.repeat(40);
    view.error = 'corpus file missing — run `npm run sync` to fetch the corpora';
  }

  try {
    const res = await fetch(WARM_START_URL);
    const text = res.ok ? await res.text() : null;
    // a missing checkpoint can still answer 200 with an html fallback page
    warmCheckpoint = text && text.trimStart().startsWith('{') ? text : null;
  } catch {
    warmCheckpoint = null;
  }
  if (!warmCheckpoint) {
    $<HTMLSelectElement>('warm').value = 'none';
    $<HTMLSelectElement>('warm').disabled = true;
  }

  startRun(true);
}

void boot();
