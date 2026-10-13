import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Tokenizer } from '../data/tokenizer.ts';
import { configFor, createModel } from '../model/factory.ts';
import { GptModel } from '../model/gpt.ts';
import { Trainer } from './trainer.ts';
import { Adam } from './optimizer.ts';
import { makeParam } from '../model/param.ts';
import { decodeCheckpoint, encodeCheckpointString } from './checkpoint.ts';

const CORPUS = 'the quick brown fox jumps over the lazy dog. '.repeat(60);

test('tokenizer round-trips text exactly', () => {
  const tok = Tokenizer.fromText(CORPUS);
  const ids = tok.encode(CORPUS);
  assert.equal(tok.decode(ids), CORPUS);
  assert.equal(tok.coverage(CORPUS), 1);
  assert.equal(tok.coverage('🎈🎈🎈'), 0, 'characters the corpus never contained are uncovered');
  assert.ok(tok.coverage('the quick fox') > 0.9, 'common English stays inside the character vocabulary');
});

test('tokenizer serialises and deserialises', () => {
  const tok = Tokenizer.fromText('hello world');
  const back = Tokenizer.fromJSON(JSON.parse(JSON.stringify(tok.toJSON())));
  assert.deepEqual(back.chars, tok.chars);
  assert.equal(back.decode(back.encode('hello')), 'hello');
});

test('learning-rate schedule warms up then decays to the floor', () => {
  const params = [makeParam('w', 4, 4, () => 0)];
  const opt = new Adam(params, { lr: 1e-2, minLr: 1e-3, warmupSteps: 10 }, 100);
  assert.ok(opt.lrAt(0) < opt.lrAt(9), 'warmup must increase the learning rate');
  assert.ok(Math.abs(opt.lrAt(9) - 1e-2) < 1e-6, 'the last warmup step reaches the peak rate');
  const late = opt.lrAt(99);
  assert.ok(late < 2e-3 && late >= 1e-3, `cosine decay should approach minLr, got ${late}`);
  assert.ok(opt.lrAt(50) > opt.lrAt(99), 'the schedule must be monotonically decreasing after warmup');
});

test('adam actually reduces a quadratic', () => {
  const params = [makeParam('x', 1, 4, () => 3)];
  const opt = new Adam(params, { lr: 0.1, warmupSteps: 0, gradClip: 0 }, 200);
  for (let i = 0; i < 200; i++) {
    for (let j = 0; j < 4; j++) params[0].grad[j] = 2 * params[0].data[j];
    opt.step();
  }
  for (let j = 0; j < 4; j++) assert.ok(Math.abs(params[0].data[j]) < 0.05, `parameter ${j} should converge to 0, got ${params[0].data[j]}`);
});

test('gradient clipping caps the update scale', () => {
  const params = [makeParam('x', 1, 4, () => 1)];
  const opt = new Adam(params, { lr: 0.1, warmupSteps: 0, gradClip: 1 }, 10);
  for (let j = 0; j < 4; j++) params[0].grad[j] = 1000;
  opt.step();
  assert.ok(opt.lastClip < 0.01, `clip factor should be tiny for huge gradients, got ${opt.lastClip}`);
});

test('training a tiny transformer reduces the loss on a patterned corpus', () => {
  const tok = Tokenizer.fromText(CORPUS);
  const data = tok.encode(CORPUS);
  const config = configFor('gpt', tok.vocabSize, { blockSize: 16, nEmbd: 32, nLayer: 1, nHead: 2, seed: 1234 });
  const model = new GptModel(config);
  const trainer = new Trainer(model, tok, data, {
    steps: 150,
    batchSize: 16,
    blockSize: 16,
    evalEvery: 50,
    evalBatches: 4,
    sampleEvery: 150,
    seed: 1234,
    optimizer: { lr: 3e-3, minLr: 3e-4, warmupSteps: 20, gradClip: 1, weightDecay: 0 },
  });

  const first = trainer.stepOnce().loss;
  let last = first;
  for (let i = 1; i < 150; i++) last = trainer.stepOnce().loss;
  const val = trainer.evaluate();

  assert.ok(Number.isFinite(trainer.totalMs) && trainer.totalMs > 0, 'training time must be measured');
  assert.ok(last < first * 0.6, `loss should fall well below the initial value: ${first.toFixed(3)} -> ${last.toFixed(3)}`);
  assert.ok(val < 3.5, `validation loss should improve too, got ${val.toFixed(3)}`);
  assert.equal(trainer.history.length, 150);
  assert.ok(trainer.history[149].valLoss !== undefined, 'the last history entry should carry the validation loss');

  const sample = trainer.sampleText('the ', 40);
  assert.equal(sample.length, 40);
  for (const ch of sample) assert.ok(tok.chars.includes(ch), `sampled out-of-vocabulary character ${JSON.stringify(ch)}`);
});

test('checkpoint round-trip preserves the exact loss', () => {
  const tok = Tokenizer.fromText(CORPUS);
  const data = tok.encode(CORPUS);
  const config = configFor('mlp', tok.vocabSize, { blockSize: 8, nEmbd: 16, nLayer: 1, nHead: 1, seed: 77 });
  const trained = createModel(config);
  const trainer = new Trainer(trained, tok, data, { steps: 20, batchSize: 8, blockSize: 8, evalEvery: 20, sampleEvery: 100, seed: 5, optimizer: { lr: 2e-3, warmupSteps: 5, weightDecay: 0 } });
  for (let i = 0; i < 20; i++) trainer.stepOnce();

  const tokens = data.subarray(0, 8);
  const targets = data.subarray(1, 9);
  const before = trained.forward(tokens, targets, 1, 8);

  const text = encodeCheckpointString(trained, tok, { name: 'round-trip', steps: 20 });
  const restored = decodeCheckpoint(text);
  const after = restored.model.forward(tokens, targets, 1, 8);

  assert.equal(after, before, 'decoded weights must reproduce the loss bit for bit');
  assert.deepEqual(restored.tokenizer.chars, tok.chars);
  assert.equal(restored.meta.steps, 20);
  assert.equal(restored.model.numParams(), trained.numParams());
});
