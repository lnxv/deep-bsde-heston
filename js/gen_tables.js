'use strict';
// Builds LaTeX tables from summary_gpu.json into ../tex/gen/*.tex
const fs = require('fs'), path = require('path');
const S = JSON.parse(fs.readFileSync(path.join(__dirname, 'summary_gpu.json'), 'utf8'));
const OUT = path.join(__dirname, 'gen');
const SIGS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], DS = [1, 2, 5, 10, 20];
const E1 = S.groups.filter(g => g.tag === 'E1');
const cell = (d, s) => E1.find(g => g.d === d && g.sigma === s);
const f = (x, n = 2) => (x === null || x === undefined || Number.isNaN(x)) ? '--' : (Math.abs(x) < 0.5 * 10 ** -n ? (0).toFixed(n) : x.toFixed(n));
const sgn = (x, n = 2) => { const s = f(x, n); return s.startsWith('-') ? '$-$' + s.slice(1) : s; };
const hdr = SIGS.map(s => s.toFixed(1)).join(' & ');
const feller = SIGS.map(s => s > 0.4 ? `\\textbf{${s.toFixed(1)}}` : s.toFixed(1)).join(' & ');

// Table 1: total error mean (sd), plus bias and solver blocks
let t = `\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{4.2pt}\n\\begin{tabular}{r${'r'.repeat(8)}}\n\\toprule\n & \\multicolumn{8}{c}{vol-of-vol $\\sv$ (bold: Feller violated, $\\sv>0.4$)}\\\\\n$d$ & ${feller}\\\\\n\\midrule\n`;
t += `\\multicolumn{9}{l}{\\emph{(a) Total error $(\\hat Y_0-u)/u$, \\%: mean over seeds (standard deviation)}}\\\\\n`;
for (const d of DS) t += `${d} & ` + SIGS.map(s => { const g = cell(d, s); return g ? `${sgn(g.tot)}\\,{\\scriptsize(${f(g.totSd)})}` : '--'; }).join(' & ') + '\\\\\n';
t += `\\midrule\n\\multicolumn{9}{l}{\\emph{(b) Discretisation bias $(u^{40}_{\\mathrm{PT}}-u)/u$, \\%, measured by Monte Carlo without any network (MC std.\\ error)}}\\\\\n`;
for (const d of DS) t += `${d} & ` + SIGS.map(s => { const g = cell(d, s); return g ? `${sgn(g.bias)}\\,{\\scriptsize(${f(g.biasSe)})}` : '--'; }).join(' & ') + '\\\\\n';
t += `\\midrule\n\\multicolumn{9}{l}{\\emph{(c) Solver error $(\\hat Y_0-u^{40}_{\\mathrm{PT}})/u$, \\%: mean over seeds}}\\\\\n`;
for (const d of DS) t += `${d} & ` + SIGS.map(s => { const g = cell(d, s); return g ? sgn(g.solver) : '--'; }).join(' & ') + '\\\\\n';
t += `\\midrule\n\\multicolumn{9}{l}{\\emph{(d) Reference price $u$ (exact for $d=1$; QE Monte Carlo for $d>1$)}}\\\\\n`;
for (const d of DS) t += `${d} & ` + SIGS.map(s => { const g = cell(d, s); return g ? f(g.truePrice, 3) : '--'; }).join(' & ') + '\\\\\n';
t += `\\bottomrule\n\\end{tabular}\n\\caption{Main experiment: Deep BSDE with the PT scheme and $N=40$ time steps, $\\kappa=2$, independent assets. Five seeds for $d\\le5$ and three for $d\\ge10$. By construction (a) $=$ (b) $+$ (c), up to the Monte Carlo error of the references.}\n\\label{tab:main}\n\\end{table}\n`;
fs.writeFileSync(path.join(OUT, 'tab_main.tex'), t);

// Table: hedging (delta error and variance reduction)
let h = `\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{4.5pt}\n\\begin{tabular}{r${'r'.repeat(8)}}\n\\toprule\n$d$ & ${feller}\\\\\n\\midrule\n\\multicolumn{9}{l}{\\emph{(a) Initial delta: mean absolute relative error vs.\\ reference delta, \\%}}\\\\\n`;
for (const d of DS) h += `${d} & ` + SIGS.map(s => { const g = cell(d, s); return g ? f(g.absDeltaErr) : '--'; }).join(' & ') + '\\\\\n';
h += `\\midrule\n\\multicolumn{9}{l}{\\emph{(b) Replication: variance reduction $1-\\mathbb E[(Y_N-g)^2]/\\mathrm{Var}[g]$ on fresh paths}}\\\\\n`;
for (const d of DS) h += `${d} & ` + SIGS.map(s => { const g = cell(d, s); return g ? f(g.hvr, 3) : '--'; }).join(' & ') + '\\\\\n';
h += `\\midrule\n\\multicolumn{9}{l}{\\emph{(c) Relative RMS replication error $\\sqrt{\\mathbb E[(Y_N-g)^2]}/\\hat Y_0$, \\%}}\\\\\n`;
for (const d of DS) h += `${d} & ` + SIGS.map(s => { const g = cell(d, s); return g ? f(g.relRMSE, 1) : '--'; }).join(' & ') + '\\\\\n';
h += `\\bottomrule\n\\end{tabular}\n\\caption{Hedge quality in the main experiment, averaged over seeds. The reference delta is analytic for $d=1$ and a pathwise Monte Carlo estimate (QE scheme) for $d>1$. Replication statistics use 8{,}192 fresh paths per run.}\n\\label{tab:hedge}\n\\end{table}\n`;
fs.writeFileSync(path.join(OUT, 'tab_hedge.tex'), h);

// Ablation table
const names = { baseline: 'Baseline ($N{=}40$, width $d{+}10$, 2 layers, 4k it., $M{=}256$)', wider: 'Wider (width $4(d{+}10)$)', deeper: 'Deeper (3 hidden layers)', moreIters: 'Longer training (10k iterations)', largerBatch: 'Larger batch ($M=1024$)', warmStart: 'Monte Carlo warm start of $Y_0$', moreSteps: 'More time steps ($N=80$)', FTscheme: 'FT scheme instead of PT' };
const order = ['baseline', 'wider', 'deeper', 'moreIters', 'largerBatch', 'warmStart', 'moreSteps', 'FTscheme'];
let a = `\\begin{table}[t]\n\\centering\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l rr rr r}\n\\toprule\n & \\multicolumn{2}{c}{$d=2$} & \\multicolumn{2}{c}{$d=10$} & $d=10,\\ \\sv{=}0.7$\\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}\nVariant & $\\sv=0.3$ & $\\sv=0.7$ & $\\sv=0.3$ & $\\sv=0.7$ & var.\\ red.\\\\\n\\midrule\n`;
for (const v of order) {
  const get = (d, s) => S.e4.find(r => r.d === d && r.s === s && r.v === v);
  const c = (d, s) => { const r = get(d, s); return r ? `${sgn(r.tot)}\\,{\\scriptsize(${f(r.totSd)})}` : '--'; };
  const vr = get(10, 0.7);
  a += `${names[v]} & ${c(2, 0.3)} & ${c(2, 0.7)} & ${c(10, 0.3)} & ${c(10, 0.7)} & ${vr ? f(vr.hvr, 3) : '--'}\\\\\n`;
}
a += `\\bottomrule\n\\end{tabular}\n\\caption{Ablations. Entries are the total pricing error in \\% (mean over 3 seeds, with the standard deviation in parentheses; the baseline row uses the 5 or 3 seeds of the main experiment). The last column is the hedge variance reduction. Only changing the time discretisation moves the pricing error systematically. Network capacity, training length and batch size improve the hedge but not the price.}\n\\label{tab:ablation}\n\\end{table}\n`;
fs.writeFileSync(path.join(OUT, 'tab_ablation.tex'), a);
console.log('tables written');
