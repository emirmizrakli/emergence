#!/usr/bin/env node
/**
 * Stage everything the demo page needs into public/.
 *
 * The page has no server: it fetches a corpus and (optionally) a checkpoint as
 * static files. This script makes sure both exist, downloading the corpus if it
 * is missing and copying the best checkpoint the CLI has produced.
 *
 *   node scripts/sync-corpora.mjs
 */
import { copyFileSync, existsSync, mkdirSync, statSync, writeFileSync, readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');

const CORPORA = [
  {
    name: 'tiny shakespeare',
    url: 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
    data: 'data/tiny-shakespeare.txt',
    public: 'public/corpora/shakespeare.txt',
    minBytes: 100_000,
  },
];

const CHECKPOINTS = [
  { from: 'checkpoints/shakespeare-gpt.json', to: 'public/checkpoints/shakespeare-gpt.json' },
];

const kb = (p) => `${(statSync(p).size / 1024).toFixed(0)} KB`;

async function ensureCorpus(entry) {
  const target = join(root, entry.data);
  const ok = existsSync(target) && statSync(target).size >= entry.minBytes;
  if (!ok) {
    process.stdout.write(`downloading ${entry.name} … `);
    const res = await fetch(entry.url);
    if (!res.ok) throw new Error(`${entry.url} → HTTP ${res.status}`);
    const text = await res.text();
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, text);
    console.log(`${(text.length / 1024).toFixed(0)} KB`);
  } else {
    console.log(`corpus present: ${entry.data} (${kb(target)})`);
  }

  const pub = join(root, entry.public);
  if (!existsSync(pub) || readFileSync(pub, 'utf8') !== readFileSync(target, 'utf8')) {
    mkdirSync(dirname(pub), { recursive: true });
    copyFileSync(target, pub);
    console.log(`staged → ${entry.public} (${kb(pub)})`);
  } else {
    console.log(`staged (up to date): ${entry.public}`);
  }
}

function stageCheckpoints() {
  let any = false;
  for (const c of CHECKPOINTS) {
    const from = join(root, c.from);
    if (!existsSync(from)) {
      console.log(`no checkpoint yet at ${c.from} — train one with \`npm run train:shakespeare\``);
      continue;
    }
    const to = join(root, c.to);
    const stale = !existsSync(to) || statSync(to).mtimeMs < statSync(from).mtimeMs;
    if (stale) {
      mkdirSync(dirname(to), { recursive: true });
      copyFileSync(from, to);
      console.log(`staged → ${c.to} (${kb(to)})`);
    } else {
      console.log(`staged (up to date): ${c.to}`);
    }
    any = true;
  }
  return any;
}

for (const entry of CORPORA) await ensureCorpus(entry);
const staged = stageCheckpoints();
if (!staged) console.log('\nthe page still works without a checkpoint: it trains from random noise.');
