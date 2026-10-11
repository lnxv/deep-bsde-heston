'use strict';
// Parallel job runner: node runner.js jobs.json results.jsonl [nWorkers]
// Resumable: skips jobs whose id already appears in results.jsonl.
const { Worker, isMainThread, parentPort, workerData } = require('worker_threads');
const fs = require('fs');
const L = require('./lib.js');

function runJob(job) {
  const t0 = Date.now(); let out;
  if (job.type === 'bsde') out = L.trainDeepBSDE({ ...job.cfg, model: job.model });
  else if (job.type === 'qe') out = L.mcQE(job.model, job.N, job.nPaths, job.seed);
  else if (job.type === 'disc') out = L.mcDiscrete(job.model, job.scheme, job.N, job.nPaths, job.seed);
  else if (job.type === 'analytic') out = L.hestonCall(job.model, job.model.K);
  else throw new Error('unknown job type ' + job.type);
  return { ...job, result: out, wallMs: Date.now() - t0 };
}

if (isMainThread) {
  const [jobsFile, outFile, nw] = process.argv.slice(2);
  const jobs = JSON.parse(fs.readFileSync(jobsFile, 'utf8'));
  const done = new Set();
  if (fs.existsSync(outFile)) for (const line of fs.readFileSync(outFile, 'utf8').split('\n')) if (line.trim()) done.add(JSON.parse(line).id);
  const queue = jobs.filter(j => !done.has(j.id));
  console.log(`total ${jobs.length}, done ${done.size}, queued ${queue.length}`);
  const W = Math.min(+(nw || 12), queue.length); let active = 0, finished = 0;
  const startNext = (w) => { const j = queue.shift(); if (!j) { w.terminate(); if (--active === 0) console.log('ALL DONE'); return; } w.postMessage(j); };
  for (let i = 0; i < W; i++) {
    const w = new Worker(__filename, { workerData: {} }); active++;
    w.on('message', (res) => { fs.appendFileSync(outFile, JSON.stringify(res) + '\n'); finished++; console.log(`[${new Date().toISOString()}] ${finished}/${queue.length + finished} ${res.id} ${(res.wallMs / 1000).toFixed(1)}s`); startNext(w); });
    w.on('error', (e) => { console.error('worker error', e); });
    startNext(w);
  }
} else {
  parentPort.on('message', (job) => { try { parentPort.postMessage(runJob(job)); } catch (e) { parentPort.postMessage({ ...job, error: String(e && e.stack || e), wallMs: 0 }); } });
}
