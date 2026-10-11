const L = require('./lib.js');
const base = { d:1, S0:100, K:100, r:0.05, T:0.5, kappa:2, theta:0.04, sigma:0.4, rho:-0.7, v0:0.04, rhoS:0 };
console.log('BS(0.2)=', L.bsCall(100,100,0.5,0.05,0.2).toFixed(5));
for (const s of [0.01, 0.1, 0.4, 0.8, 1.0]) {
  const p = {...base, sigma:s};
  const a = L.hestonCall(p, 100), b = L.hestonCall(p, 100, 1000, 100000);
  console.log('sig',s,'analytic',a.price.toFixed(6), 'U1000', b.price.toFixed(6), 'delta', a.delta.toFixed(5));
}
let t=Date.now();
for (const s of [0.4, 0.8]) {
  const p = {...base, sigma:s};
  const q = L.mcQE(p, 50, 2_000_000, 1);
  const a = L.hestonCall(p,100);
  console.log('QE sig',s, q.price.toFixed(4),'+-',q.se.toFixed(4),' analytic',a.price.toFixed(4),' delta', q.delta.toFixed(4),'+-',q.deltaSe.toFixed(4),'vs',a.delta.toFixed(4), 'ms',Date.now()-t); t=Date.now();
  const e = L.mcDiscrete(p, 'PT', 40, 2_000_000, 2);
  console.log('  Euler-PT N40', e.price.toFixed(4),'+-',e.se.toFixed(4), 'ms',Date.now()-t); t=Date.now();
  const f = L.mcDiscrete(p, 'FT', 40, 2_000_000, 3);
  console.log('  FT N40', f.price.toFixed(4),'+-',f.se.toFixed(4), 'ms',Date.now()-t); t=Date.now();
}
const gc = L.trainDeepBSDE({ model:{...base, d:3, sigma:0.6}, seed:1, gradCheck:true, N:10, batch:64 });
for (const r of gc) console.log('grad', r.k, r.analytic.toExponential(4), r.fd.toExponential(4));
