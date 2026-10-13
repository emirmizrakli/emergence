/**
 * Headless training CLI.
 *
 *   node src/cli/train.ts --corpus data/tiny-shakespeare.txt --steps 2000
 *   node src/cli/train.ts --compare --steps 600 --corpus data/tiny-shakespeare.txt
 *   node src/cli/train.ts --model gpt --steps 4000 --embd 128 --layers 4 \
 *     --out checkpoints/shakespeare-gpt.json
 *
 * Everything the browser demo does, minus the pixels — which is what makes the
 * numbers in the README reproducible on any machine with Node 26+ (native
 * TypeScript, no build step).
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { basename, dirname } from 'node:path';
import { Tokenizer } from '../data/tokenizer.ts';
import { configFor, createModel } from '../model/factory.ts';
import type { ModelKind } from '../model/base.ts';
import { Trainer, type TrainConfig } from '../train/trainer.ts';
import { encodeCheckpoint } from '../train/checkpoint.ts';

type FlagValue = string | number | boolean | undefined;

interface Args {
  [key: string]: FlagValue;
}

function parseArgs(argv: string[]): Args {
  const out: Args = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith('--')) {
      out[key] = true;
    } else {
      out[key] = next;
      i++;
    }
  }
  return out;
}

const num = (v: FlagValue, fallback: number): number => (v === undefined || v === true ? fallback : Number(v));
const str = (v: FlagValue, fallback: string): string => (typeof v === 'string' ? v : fallback);

function bar(value: number, max: number, width = 30): string {
  const filled = Math.max(0, Math.min(width, Math.round((value / max) * width)));
  return '█'.repeat(filled) + '·'.repeat(width - filled);
}

const fmt = (n: number, digits = 4): string => (Number.isFinite(n) ? n.toFixed(digits) : 'n/a');

interface RunResult {
  kind: ModelKind;
  params: number;
  firstLoss: number;
  finalLoss: number;
  valLoss: number;
  minutes: number;
  tokensPerSec: number;
  samples: string[];
  checkpointPath: string | null;
  perplexity: number;
}

function trainOne(kind: ModelKind, corpusPath: string, opts: Args): RunResult {
  const text = readFileSync(corpusPath, 'utf8');
  const tokenizer = Tokenizer.fromText(text);
  const data = tokenizer.encode(text);
  const vocabSize = tokenizer.vocabSize;

  const steps = num(opts.steps, 1000);
  const blockSize = num(opts.block, 32);
  const batchSize = num(opts.batch, 32);
  const nEmbd = num(opts.embd, kind === 'gpt' ? 64 : 64);
  const nLayer = num(opts.layers, kind === 'gpt' ? 2 : 1);
  const nHead = num(opts.heads, 4);
  const lr = num(opts.lr, 3e-3);
  const seed = num(opts.seed, 1337);
  const evalEvery = num(opts.eval, Math.max(25, Math.floor(steps / 8)));
  const sampleEvery = num(opts['sample-every'], Math.max(100, Math.floor(steps / 3)));

  const config = configFor(kind, vocabSize, { blockSize, nEmbd, nLayer, nHead, seed });
  const model = createModel(config);
  const trainCfg: Partial<TrainConfig> = {
    steps,
    batchSize,
    blockSize,
    evalEvery,
    sampleEvery,
    temperature: num(opts.temperature, 0.8),
    topK: num(opts.topk, 40),
    seed,
    optimizer: {
      lr,
      minLr: lr / 10,
      warmupSteps: Math.max(5, Math.floor(steps * 0.02)),
      gradClip: num(opts.clip, 1),
      weightDecay: num(opts.wd, 0.01),
    },
  };
  const trainer = new Trainer(model, tokenizer, data, trainCfg);

  console.log(`\n=== ${kind.toUpperCase()} — ${basename(corpusPath)} ===`);
  console.log(
    `params=${model.numParams().toLocaleString('en-US')} vocab=${vocabSize} block=${blockSize} batch=${batchSize} lr=${lr} steps=${steps}`,
  );
  console.log(`train tokens=${trainer.trainTokens.toLocaleString('en-US')} val tokens=${trainer.valTokens.toLocaleString('en-US')}`);

  const samples: string[] = [];
  let firstLoss = Number.NaN;
  const t0 = Date.now();
  for (let s = 0; s < steps; s++) {
    const report = trainer.stepOnce();
    if (s === 0) firstLoss = report.loss;
    if ((s + 1) % evalEvery === 0 || s === steps - 1) {
      const val = trainer.evaluate();
      console.log(
        `step ${String(report.step).padStart(5)} | loss ${fmt(report.loss)} | val ${fmt(val)} | lr ${report.lr.toExponential(2)} | ${bar(Math.min(1, report.loss / 5), 1)} | ${Math.round(report.tokensPerSec).toLocaleString('en-US')} tok/s`,
      );
    }
    if ((s + 1) % sampleEvery === 0 || s === steps - 1) {
      const textOut = trainer.sampleText().replace(/\n/g, ' ⏎ ');
      samples.push(textOut);
      console.log(`  sample @${report.step}: ${textOut.slice(0, 220)}`);
    }
  }

  const minutes = (Date.now() - t0) / 60000;
  const finalLoss = trainer.history[trainer.history.length - 1]?.loss ?? Number.NaN;
  const valLoss = trainer.lastValLoss ?? Number.NaN;
  const tokensPerSec = (batchSize * blockSize * steps) / (trainer.totalMs / 1000);

  let checkpointPath: string | null = null;
  const outArg = str(opts.out, '');
  if (outArg && !opts.compare) {
    mkdirSync(dirname(outArg), { recursive: true });
    const checkpoint = encodeCheckpoint(model, tokenizer, {
      name: `${kind}-${basename(corpusPath)}`,
      corpus: basename(corpusPath),
      steps,
      loss: finalLoss,
      valLoss,
      notes: `node src/cli/train.ts --model ${kind} --corpus ${basename(corpusPath)} --steps ${steps}`,
    });
    const json = JSON.stringify(checkpoint);
    writeFileSync(outArg, json);
    checkpointPath = outArg;
    console.log(`  checkpoint → ${outArg} (${(json.length / 1024).toFixed(0)} KB)`);
  }

  return {
    kind,
    params: model.numParams(),
    firstLoss,
    finalLoss,
    valLoss,
    minutes,
    tokensPerSec,
    samples,
    checkpointPath,
    perplexity: Math.exp(valLoss),
  };
}

function main(): void {
  const opts = parseArgs(process.argv.slice(2));
  const corpus = str(opts.corpus, 'data/tiny-shakespeare.txt');

  if (opts.compare) {
    const steps = num(opts.steps, 600);
    console.log(`corpus: ${corpus}  |  steps per model: ${steps}`);
    const kinds: ModelKind[] = ['bigram', 'mlp', 'gpt'];
    const rows = kinds.map((k) =>
      trainOne(k, corpus, { ...opts, steps, batch: num(opts.batch, 32), block: num(opts.block, 32), embd: num(opts.embd, 64), layers: num(opts.layers, 2), heads: num(opts.heads, 4) }),
    );
    console.log('\n\n=== comparison (identical corpus, identical budget) ===');
    console.log('| model | params | first loss | final loss | val loss | val perplexity | minutes | tok/s |');
    console.log('| --- | --- | --- | --- | --- | --- | --- | --- |');
    for (const r of rows) {
      console.log(
        `| ${r.kind} | ${r.params.toLocaleString('en-US')} | ${fmt(r.firstLoss, 3)} | ${fmt(r.finalLoss, 3)} | ${fmt(r.valLoss, 3)} | ${r.perplexity.toFixed(1)} | ${r.minutes.toFixed(1)} | ${Math.round(r.tokensPerSec).toLocaleString('en-US')} |`,
      );
    }
    console.log(`\ncounts: ${rows.length} models trained`);
    return;
  }

  const kind = str(opts.model, 'gpt') as ModelKind;
  trainOne(kind, corpus, opts);
}

main();
