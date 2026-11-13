'use strict';
const fs = require('node:fs'), vm = require('node:vm'), assert = require('node:assert/strict');
const site = 'dist/', context = vm.createContext({window: {}});
vm.runInContext(fs.readFileSync(site + 'GTM_Model0_scene_review.js', 'utf8'), context);
const {category} = context.window.GTMSceneReview;
const inventory = JSON.parse(fs.readFileSync(site + 'data/v11-predictions.json', 'utf8'));
const catalog = JSON.parse(fs.readFileSync(site + 'data/research-catalog.json', 'utf8'));
const run = catalog.experiments.find(e => e.id === inventory.runId);
const lookup = new Map(run.scenes.map(s => [s.id, s]));
assert.equal(inventory.total, 198); assert.equal(lookup.size, 198);
assert.equal(new Set(inventory.predictions.map(s => s.id)).size, 198);
const expected = {strong: 13, partial: 10, weak: 4, miss: 19, false: 18, clear: 134}, counts = {};
const sums = {tp: 0, fp: 0, fn: 0, tn: 0};
for (const row of inventory.predictions) {
  const source = lookup.get(row.id);
  assert.ok(source); assert.equal(category(source), row.category);
  counts[row.category] = (counts[row.category] || 0) + 1;
  for (const key of Object.keys(sums)) { assert.equal(row[key], source.nativeMetrics[key]); sums[key] += row[key]; }
  const url = new URL(row.viewerUrl, 'https://preview.test/');
  assert.equal(url.searchParams.get('scene'), row.id);
  assert.equal(url.searchParams.get('experiment'), run.id);
  assert.equal(url.searchParams.get('layout'), 'panels'); assert.equal(url.hash, '#compare');
  for (const key of ['rgb', 'truthImage', 'predictionImage', 'disagreementImage']) {
    assert.equal(row[key], source[key]); assert.ok(fs.existsSync(site + row[key]));
  }
}
assert.deepEqual(counts, expected);
for (const key of Object.keys(sums)) assert.equal(sums[key], run.nativeAggregate[key]);
const fixture = (tp, fp, fn) => ({nativeMetrics: {tp, fp, fn, tn: 100}});
assert.equal(category(fixture(5, 5, 0)), 'strong'); // exactly 50% shared area
assert.equal(category(fixture(1, 9, 0)), 'partial'); // exactly 10%
assert.equal(category(fixture(1, 10, 0)), 'weak');
assert.equal(category(fixture(0, 100, 1)), 'miss'); // wrong marks are not plume recovery
assert.equal(category(fixture(0, 1, 0)), 'false');
assert.equal(category(fixture(0, 0, 0)), 'clear');
assert.equal(category({}), 'unknown');
for (const s of ['index.html', 'progress.html']) {
  const html = fs.readFileSync(site + s, 'utf8');
  for (const m of html.matchAll(/(?:href|src)="([^"#]+)"/g)) {
    if (/^(https?:|data:)/.test(m[1])) continue;
    assert.ok(fs.existsSync(site + m[1].split(/[?#]/)[0]), `${s}: ${m[1]}`);
  }
}
console.log('PASS: all 198 predictions, category boundaries, source counts, image paths and three-pane links.');
