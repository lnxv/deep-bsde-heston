'use strict';
// Generates jobs.json for all experiments.
const fs = require('fs');
const base = { S0: 100, K: 100, r: 0.05, T: 0.5, kappa: 2, theta: 0.04, rho: -0.7, v0: 0.04, rhoS: 0 };
const jobs = [];
const r2 = (x) => Math.round(x * 100) / 100;
const pathsFor = (d) => d <= 2 ? 4e6 : d <= 5 ? 2e6 : 1e6;
const costBSDE = (d, cfg) => (cfg.iters || 4000) * (cfg.N || 40) * (cfg.batch || 256) * Math.pow((cfg.width || d + 10), 2) * (cfg.depth || 2) * 1e-9 + (cfg.iters || 4000) * (cfg.N || 40) * d * 1e-6;
const addRef = (id, model, kinds) => {
  if (kinds.includes('true')) {
    if (model.d === 1 && !model.rhoS) jobs.push({ id: id + '|analytic', type: 'analytic', model, cost: 1 });
    if (model.d > 1 && !model.rhoS) jobs.push({ id: id + '|qe', type: 'qe', model, N: 50, nPaths: pathsFor(model.d), seed: 101, cost: pathsFor(model.d) * 50 * model.d * 1e-7 });
    if (model.d === 1 && !model.rhoS) jobs.push({ id: id + '|qe', type: 'qe', model, N: 50, nPaths: 2e6, seed: 101, cost: 10 });
    if (model.rhoS) jobs.push({ id: id + '|fine', type: 'disc', model, scheme: 'FT', N: 400, nPaths: model.d >= 10 ? 4e5 : 1e6, seed: 103, cost: 400 * model.d * 4e5 * 1e-7 });
  }
  for (const k of kinds) { const m = /^disc:(PT|FT):(\d+)$/.exec(k); if (m) jobs.push({ id: `${id}|disc:${m[1]}:${m[2]}`, type: 'disc', model, scheme: m[1], N: +m[2], nPaths: pathsFor(model.d), seed: 102 + +m[2], cost: pathsFor(model.d) * +m[2] * model.d * 1e-7 }); }
};
const addBSDE = (id, model, cfg, seed) => jobs.push({ id: `${id}|bsde|${JSON.stringify(cfg)}|s${seed}`, type: 'bsde', model, cfg: { ...cfg, seed }, cost: costBSDE(model.d, cfg) });
const mk = (o) => ({ ...base, ...o });
const mid = (m) => `d${m.d}_k${m.kappa}_th${m.theta}_sig${m.sigma}_rS${m.rhoS}`;

// E1: main grid
for (const d of [1, 2, 5, 10, 20]) for (let s = 1; s <= 8; s++) {
  const m = mk({ d, sigma: r2(s / 10) });
  addRef(mid(m), m, ['true', 'disc:PT:40']);
  const nSeeds = d <= 5 ? 5 : 3;
  for (let seed = 1; seed <= nSeeds; seed++) addBSDE(mid(m), m, { tag: 'E1' }, seed);
}
// E2: Feller shift via kappa (MC-only bias maps + some BSDE runs)
for (const kappa of [1, 2, 4]) for (let s = 2; s <= 20; s++) {
  const m = mk({ d: 1, kappa, sigma: r2(s * 0.05) });
  addRef(mid(m), m, ['true', 'disc:PT:40', 'disc:FT:40']);
}
for (const kappa of [1, 2, 4]) for (let s = 1; s <= 10; s++) {
  const m = mk({ d: 5, kappa, sigma: r2(s / 10) });
  addRef(mid(m), m, ['true', 'disc:PT:40', 'disc:FT:40']);
}
for (const kappa of [1, 4]) for (const sg of [0.2, 0.4, 0.6, 0.8, 1.0]) {
  const m = mk({ d: 1, kappa, sigma: sg });
  for (let seed = 1; seed <= 2; seed++) addBSDE(mid(m), m, { tag: 'E2' }, seed);
}
// E3: time steps & scheme
for (const d of [1, 5]) for (const sg of [0.3, 0.5, 0.7]) {
  const m = mk({ d, sigma: sg });
  const kinds = ['true']; for (const sc of ['PT', 'FT']) for (const N of [10, 20, 40, 80, 160]) kinds.push(`disc:${sc}:${N}`);
  addRef(mid(m), m, kinds);
  for (const sc of ['PT', 'FT']) for (const N of (d === 1 ? [10, 20, 40, 80] : [40, 80])) if (!(sc === 'PT' && N === 40)) addBSDE(mid(m), m, { tag: 'E3', scheme: sc, N }, 1);
}
// E4: ablations (d=2 and d=10, sigma 0.3 / 0.7)
for (const d of [2, 10]) for (const sg of [0.3, 0.7]) {
  const m = mk({ d, sigma: sg });
  addRef(mid(m), m, ['true', 'disc:PT:40', 'disc:PT:80']);
  const variants = [{ tag: 'E4', variant: 'wider', width: d === 2 ? 48 : 40 }, { tag: 'E4', variant: 'deeper', depth: 3 },
    { tag: 'E4', variant: 'moreIters', iters: 10000 }, { tag: 'E4', variant: 'largerBatch', batch: 1024 },
    { tag: 'E4', variant: 'moreSteps', N: 80 }, { tag: 'E4', variant: 'warmStart', warmStart: true }];
  const seeds = d === 2 ? [1, 2] : [1];
  for (const v of variants) for (const seed of seeds) addBSDE(mid(m), m, v, seed);
}
// E5: correlated assets rhoS=0.5
for (const d of [2, 5, 10, 20]) for (const sg of [0.3, 0.7]) {
  const m = mk({ d, sigma: sg, rhoS: 0.5 });
  addRef(mid(m), m, ['true', 'disc:PT:40']);
  for (let seed = 1; seed <= 2; seed++) addBSDE(mid(m), m, { tag: 'E5' }, seed);
}
// check fine-FT reference bias at d=1 (vs analytic) for E5 validity
for (const sg of [0.3, 0.7]) { const m = mk({ d: 1, sigma: sg }); jobs.push({ id: mid(m) + '|fineFT400', type: 'disc', model: m, scheme: 'FT', N: 400, nPaths: 2e6, seed: 104, cost: 80 }); }

// dedupe by id, sort by cost desc
const seen = new Map(); for (const j of jobs) { j.model.d = j.model.d; if (!seen.has(j.id)) seen.set(j.id, j); }
const out = [...seen.values()].sort((a, b) => b.cost - a.cost);
fs.writeFileSync('jobs.json', JSON.stringify(out));
const nb = out.filter(j => j.type === 'bsde').length;
console.log('jobs', out.length, 'bsde', nb, 'total cost units', out.reduce((a, j) => a + j.cost, 0).toFixed(0));
