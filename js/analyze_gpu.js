'use strict';
// Aggregates GPU results (results_gpu.jsonl) -> console summary + pgfplots .dat files in ../tex/gen + summary_gpu.json
const fs = require('fs'), path = require('path');
const OUT = path.join(__dirname, 'gen'); fs.mkdirSync(OUT, { recursive: true });
const rows = fs.readFileSync(path.join(__dirname, 'results_gpu.jsonl'), 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const analytic = JSON.parse(fs.readFileSync(path.join(__dirname, 'analytic.json'), 'utf8'));
const key = (m) => `d${m.d}_k${+m.kappa}_s${+m.sigma}_r${+m.rhoS}`;
const refs = {};
for (const r of rows.filter(r => r.kind === 'ref')) {
  const k = key(r.model); refs[k] = refs[k] || {};
  const prev = refs[k][r.ref], x = r.result;
  if (!prev) refs[k][r.ref] = { ...x, sec: r.stackSec / r.R };
  else { // pool independent estimates (inverse variance)
    const w1 = 1 / prev.se ** 2, w2 = 1 / x.se ** 2, wd1 = 1 / prev.deltaSe ** 2, wd2 = 1 / x.deltaSe ** 2;
    refs[k][r.ref] = { ...prev, price: (prev.price * w1 + x.price * w2) / (w1 + w2), se: Math.sqrt(1 / (w1 + w2)), delta: (prev.delta * wd1 + x.delta * wd2) / (wd1 + wd2), deltaSe: Math.sqrt(1 / (wd1 + wd2)), nPaths: prev.nPaths + x.nPaths };
  }
}
const truth = (m) => {
  if (m.rhoS) { const R = (refs[key(m)] || {})['disc:FT:400']; return R && { price: R.price, se: R.se, delta: R.delta, src: 'FT400' }; }
  if (m.d === 1) { const a = analytic[`${+m.kappa}_${+m.sigma}`]; return a && { price: a.price, se: 0, delta: a.delta, src: 'analytic' }; }
  const R = (refs[key(m)] || {}).qe; return R && { price: R.price, se: R.se, delta: R.delta, src: 'QE' };
};
const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
const sd = (a) => { if (a.length < 2) return 0; const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / (a.length - 1)); };
const f = (x, n = 3) => (x === undefined || x === null || Number.isNaN(x)) ? 'nan' : x.toFixed(n);
const groups = {};
for (const r of rows.filter(r => r.kind === 'bsde')) { const gk = r.tag + '|' + key(r.model) + '|' + JSON.stringify(r.cfg); (groups[gk] = groups[gk] || { tag: r.tag, model: r.model, cfg: r.cfg, runs: [] }).runs.push(r); }
function summarize(g) {
  const m = g.model, T = truth(m), scheme = g.cfg.scheme || 'PT', N = g.cfg.N || 40;
  const disc = (refs[key(m)] || {})[`disc:${scheme}:${N}`];
  const P = g.runs.map(r => r.result.price), Dl = g.runs.map(r => r.result.delta);
  const tot = T ? P.map(p => 100 * (p - T.price) / T.price) : [], sol = disc && T ? P.map(p => 100 * (p - disc.price) / T.price) : [];
  return {
    tag: g.tag, d: m.d, sigma: +m.sigma, kappa: +m.kappa, rhoS: +m.rhoS, cfg: g.cfg, n: g.runs.length, scheme, N,
    truePrice: T && T.price, trueSe: T && T.se, trueSrc: T && T.src, discPrice: disc && disc.price, discSe: disc && disc.se,
    priceMean: mean(P), priceSd: sd(P),
    tot: tot.length ? mean(tot) : null, totSd: tot.length ? sd(tot) : null, absTot: tot.length ? mean(tot.map(Math.abs)) : null,
    bias: (T && disc) ? 100 * (disc.price - T.price) / T.price : null, biasSe: (T && disc) ? 100 * Math.hypot(disc.se, T.se) / T.price : null,
    solver: sol.length ? mean(sol) : null, solverSd: sol.length ? sd(sol) : null, absSolver: sol.length ? mean(sol.map(Math.abs)) : null,
    deltaMean: mean(Dl), deltaSd: sd(Dl), trueDelta: T && T.delta, discDelta: disc && disc.delta,
    deltaErr: (T && T.delta) ? mean(Dl.map(x => 100 * (x - T.delta) / T.delta)) : null,
    absDeltaErr: (T && T.delta) ? mean(Dl.map(x => 100 * Math.abs(x - T.delta) / T.delta)) : null,
    absDeltaErrDisc: disc ? mean(Dl.map(x => 100 * Math.abs(x - disc.delta) / disc.delta)) : null,
    hvr: mean(g.runs.map(r => r.result.hedgeVarReduction)), relRMSE: mean(g.runs.map(r => 100 * Math.sqrt(r.result.valMSE) / r.result.price)),
    valMSE: mean(g.runs.map(r => r.result.valMSE)), payoffVar: mean(g.runs.map(r => r.result.payoffVar)),
    stackSec: g.runs[0].result.trainSecStack, R: g.runs[0].result.R, nParams: g.runs[0].result.nParams,
  };
}
const all = Object.values(groups).map(summarize);
const S = { groups: all, refs };
const wr = (name, header, lines) => fs.writeFileSync(path.join(OUT, name), header + '\n' + lines.join('\n') + '\n');
const SIGS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], DS = [1, 2, 5, 10, 20];
const E1 = all.filter(g => g.tag === 'E1');
const cell = (d, s) => E1.find(g => g.d === d && g.sigma === s);
const pg = (name, fn, n = 2) => { console.log('\n' + name + '\n  d | ' + SIGS.join('    ')); for (const d of DS) console.log(String(d).padStart(3) + ' | ' + SIGS.map(s => { const g = cell(d, s); return g ? f(fn(g), n).padStart(6) : '   -- '; }).join(' ')); };
pg('E1 total err % (mean signed)', g => g.tot); pg('E1 total err sd %', g => g.totSd); pg('E1 |total| %', g => g.absTot);
pg('E1 disc bias % (PT40-true)', g => g.bias); pg('E1 bias se %', g => g.biasSe, 3);
pg('E1 solver err % (BSDE-PT40)/true', g => g.solver); pg('E1 solver sd %', g => g.solverSd); pg('E1 |solver| %', g => g.absSolver);
pg('E1 true price', g => g.truePrice, 4); pg('E1 true se', g => g.trueSe, 5);
pg('E1 delta err % (signed vs true)', g => g.deltaErr); pg('E1 |delta err| % vs true', g => g.absDeltaErr); pg('E1 |delta err| % vs disc', g => g.absDeltaErrDisc);
pg('E1 hedge var reduction', g => g.hvr, 4); pg('E1 rel RMSE replication %', g => g.relRMSE, 1); pg('E1 val MSE', g => g.valMSE, 3);
for (const d of DS) wr(`e1_d${d}.dat`, 'sigma tot totsd bias biasse solver solversd deltaerr absdeltaerr hvr relrmse', E1.filter(g => g.d === d).sort((a, b) => a.sigma - b.sigma).map(g => [g.sigma, g.tot, g.totSd, g.bias, g.biasSe, g.solver, g.solverSd, g.deltaErr, g.absDeltaErr, g.hvr, g.relRMSE].map(x => f(x, 4)).join(' ')));
// by-d files at fixed sigma
for (const s of SIGS) wr(`e1_bys${s}.dat`, 'd tot totsd bias biasse solver solversd hvr relrmse absdeltaerr', DS.map(d => cell(d, s)).filter(Boolean).map(g => [g.d, g.tot, g.totSd, g.bias, g.biasSe, g.solver, g.solverSd, g.hvr, g.relRMSE, g.absDeltaErr].map(x => f(x, 4)).join(' ')));

// E2 bias maps
const e2 = [];
for (const [k, R] of Object.entries(refs)) {
  const m = /^d(\d+)_k([\d.]+)_s([\d.]+)_r0$/.exec(k); if (!m) continue;
  const mm = { d: +m[1], kappa: +m[2], sigma: +m[3], rhoS: 0 }; const T = truth(mm); if (!T) continue;
  const row = { ...mm, feller: 2 * mm.kappa * 0.04 / mm.sigma ** 2, truePrice: T.price };
  for (const rk of Object.keys(R)) if (rk.startsWith('disc:')) { row[rk] = 100 * (R[rk].price - T.price) / T.price; row[rk + ':se'] = 100 * Math.hypot(R[rk].se, T.se) / T.price; }
  if (mm.d === 1 && R.qe) { row.qeChk = 100 * (R.qe.price - T.price) / T.price; row.qeChkSe = 100 * R.qe.se / T.price; row.qeDeltaChk = R.qe.delta - T.delta; }
  e2.push(row);
}
S.e2 = e2;
for (const d of [1, 5]) for (const kappa of [1, 2, 4]) {
  const rr = e2.filter(r => r.d === d && r.kappa === kappa && r['disc:PT:40'] !== undefined && r['disc:FT:40'] !== undefined).sort((a, b) => a.sigma - b.sigma);
  wr(`e2_d${d}_k${kappa}.dat`, 'sigma feller invfeller pt ptse ft ftse', rr.map(r => [r.sigma, r.feller, 1 / r.feller, r['disc:PT:40'], r['disc:PT:40:se'], r['disc:FT:40'], r['disc:FT:40:se']].map(x => f(x, 4)).join(' ')));
  wr(`e2n_d${d}_k${kappa}.dat`, 'feller nb', rr.filter(r => r.feller <= 1.6).map(r => [r.feller, r['disc:PT:40'] / (r.sigma * r.sigma)].map(x => f(x, 4)).join(' ')));
  console.log(`\nE2 bias d=${d} k=${kappa}: sigma feller PT40% FT40%`); for (const r of rr) console.log(' ', r.sigma, f(r.feller, 2), f(r['disc:PT:40'], 3), '±' + f(r['disc:PT:40:se'], 3), f(r['disc:FT:40'], 3));
}
const qc = e2.filter(r => r.qeChk !== undefined); console.log('\nQE vs analytic d=1: max |z| =', f(Math.max(...qc.map(r => Math.abs(r.qeChk / r.qeChkSe))), 2), ' max |rel err| % =', f(Math.max(...qc.map(r => Math.abs(r.qeChk))), 4), ' n =', qc.length, ' max |delta diff| =', f(Math.max(...qc.map(r => Math.abs(r.qeDeltaChk))), 5));
S.qeCheck = qc.map(r => ({ kappa: r.kappa, sigma: r.sigma, err: r.qeChk, se: r.qeChkSe }));
const E2b = all.filter(g => (g.tag === 'E2' || g.tag === 'E1') && g.d === 1);
console.log('\nE2 BSDE d=1: kappa sigma feller tot% bias% solver% (sd)');
for (const g of E2b.sort((a, b) => a.kappa - b.kappa || a.sigma - b.sigma)) console.log(' ', g.kappa, g.sigma, f(2 * g.kappa * 0.04 / g.sigma ** 2, 2), f(g.tot), f(g.bias), f(g.solver), '±' + f(g.solverSd));
for (const kappa of [1, 2, 4]) wr(`e2bsde_k${kappa}.dat`, 'sigma feller invfeller tot totsd bias solver solversd', E2b.filter(g => g.kappa === kappa).sort((a, b) => a.sigma - b.sigma).map(g => [g.sigma, 2 * kappa * 0.04 / g.sigma ** 2, g.sigma ** 2 / (2 * kappa * 0.04), g.tot, g.totSd, g.bias, g.solver, g.solverSd].map(x => f(x, 4)).join(' ')));

// E3
console.log('\nE3 MC bias vs N (%):');
for (const d of [1, 5]) for (const s of [0.3, 0.5, 0.7]) for (const sc of ['PT', 'FT']) {
  const r = e2.find(x => x.d === d && x.kappa === 2 && x.sigma === s); if (!r) continue;
  const pts = [10, 20, 40, 80, 160].map(N => [N, r[`disc:${sc}:${N}`], r[`disc:${sc}:${N}:se`]]).filter(p => p[1] !== undefined);
  console.log(' ', d, s, sc, pts.map(p => `${p[0]}:${f(p[1], 3)}`).join(' '));
  wr(`e3_d${d}_s${s}_${sc}.dat`, 'N bias se', pts.map(p => p.map(x => f(x, 4)).join(' ')));
}
const E3 = all.filter(g => (g.tag === 'E3' || g.tag === 'E1') && [1, 5].includes(g.d) && [0.3, 0.5, 0.7].includes(g.sigma));
console.log('\nE3 BSDE: d sigma scheme N tot% (sd) bias% solver%');
for (const g of E3.sort((a, b) => a.d - b.d || a.sigma - b.sigma || a.scheme.localeCompare(b.scheme) || a.N - b.N)) console.log(' ', g.d, g.sigma, g.scheme, g.N, f(g.tot), '±' + f(g.totSd), f(g.bias), f(g.solver), 'n=' + g.n);
for (const d of [1, 5]) for (const s of [0.3, 0.5, 0.7]) for (const sc of ['PT', 'FT']) wr(`e3bsde_d${d}_s${s}_${sc}.dat`, 'N tot totsd bias solver', E3.filter(g => g.d === d && g.sigma === s && g.scheme === sc).sort((a, b) => a.N - b.N).map(g => [g.N, g.tot, g.totSd, g.bias, g.solver].map(x => f(x, 4)).join(' ')));

// E4
console.log('\nE4: d sigma variant tot% (sd) |tot| solver% (sd) hvr n');
const E4 = [];
for (const d of [2, 10]) for (const s of [0.3, 0.7]) {
  const base = E1.find(g => g.d === d && g.sigma === s);
  const list = [['baseline', base], ...all.filter(g => g.tag === 'E4' && g.d === d && g.sigma === s).map(g => [g.cfg.variant, g])];
  for (const [v, g] of list) { if (!g) continue; console.log(' ', d, s, v.padEnd(12), f(g.tot), '±' + f(g.totSd), f(g.absTot), f(g.solver), '±' + f(g.solverSd), f(g.hvr, 4), g.n); E4.push({ d, s, v, tot: g.tot, totSd: g.totSd, absTot: g.absTot, solver: g.solver, solverSd: g.solverSd, hvr: g.hvr, n: g.n, bias: g.bias }); }
}
S.e4 = E4;
// E5
console.log('\nE5 rhoS=0.5: d sigma true(src) tot% bias% solver% hvr');
const E5 = all.filter(g => g.tag === 'E5');
for (const g of E5.sort((a, b) => a.sigma - b.sigma || a.d - b.d)) console.log(' ', g.d, g.sigma, f(g.truePrice, 4), g.trueSrc, f(g.tot), f(g.bias), f(g.solver), f(g.hvr, 4));
for (const s of [0.3, 0.7]) wr(`e5_s${s}.dat`, 'd tot totsd bias solver hvr', E5.filter(g => g.sigma === s).sort((a, b) => a.d - b.d).map(g => [g.d, g.tot, g.totSd, g.bias, g.solver, g.hvr].map(x => f(x, 4)).join(' ')));
const fine1 = [0.3, 0.7].map(s => { const R = refs[key({ d: 1, kappa: 2, sigma: s, rhoS: 0 })] || {}; const a = analytic[`2_${s}`]; return R['disc:FT:400'] ? `sigma ${s}: FT400 ${f(R['disc:FT:400'].price, 4)}±${f(R['disc:FT:400'].se, 4)} analytic ${f(a.price, 4)} (${f(100 * (R['disc:FT:400'].price - a.price) / a.price, 3)}%)` : ''; });
console.log('\nFT400 check:', fine1.join(' | '));
// timing
console.log('\nTIME (GPU, single run, s):', all.filter(g => g.tag === 'TIME').sort((a, b) => a.d - b.d).map(g => `d${g.d}: ${f(g.stackSec, 1)}s params ${g.nParams}`).join(' | '));
console.log('E1 stack secs:', E1.filter(g => g.sigma === 0.1).map(g => `d${g.d}: ${f(g.stackSec, 0)}s R=${g.R}`).join(' | '));
wr('e1_params.dat', 'd params', DS.map(d => cell(d, 0.4)).filter(Boolean).map(g => `${g.d} ${g.nParams}`));
fs.writeFileSync(path.join(__dirname, 'summary_gpu.json'), JSON.stringify(S, null, 1));
console.log('\nbsde groups', all.length, 'bsde runs', rows.filter(r => r.kind === 'bsde').length, 'refs rows', rows.filter(r => r.kind === 'ref').length);
