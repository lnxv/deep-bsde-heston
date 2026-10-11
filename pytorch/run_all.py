# Experiment driver (run after dbsde_torch definitions are loaded)
import json, time, math

OUT = open('results_gpu.jsonl', 'a')
def emit(kind, payload):
    line = json.dumps({'kind': kind, **payload})
    OUT.write(line + '\n'); OUT.flush()
    print('RES ' + line, flush=True)

def M(d, sigma, kappa=2.0, rhoS=0.0):
    return dict(d=d, sigma=round(sigma, 4), kappa=kappa, rhoS=rhoS)

def paths_for(d):
    return 8_000_000 if d <= 2 else (4_000_000 if d <= 5 else 2_000_000)

def ref(kind, models, **kw):
    t = time.time()
    if kind == 'qe':
        res = mc_qe(models, kw.get('N', 50), kw.get('nPaths', paths_for(models[0]['d'])), kw.get('seed', 101))
    else:
        res = mc_disc(models, kw['scheme'], kw['N'], kw.get('nPaths', paths_for(models[0]['d'])), kw.get('seed', 102 + kw['N']))
    dt = time.time() - t
    for m, r in zip(models, res):
        emit('ref', dict(ref=kind if kind == 'qe' else f"disc:{kw['scheme']}:{kw['N']}", model=m, result=r, stackSec=dt, R=len(models)))

def bsde(tag, models, seeds, cfg):
    t = time.time()
    res = train_stack(models, seeds, cfg)
    for m, s, r in zip(models, seeds, res):
        emit('bsde', dict(tag=tag, model=m, seed=s, cfg=cfg, result=r))
    print(f'# stack {tag} d={models[0]["d"]} R={len(models)} cfg={cfg} took {time.time()-t:.1f}s', flush=True)

SIGS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
T0 = time.time()
# ---------------- references ----------------
for d in [1, 2, 5, 10, 20]:
    ms = [M(d, s) for s in SIGS]
    if d > 1: ref('qe', ms)
    ref('disc', ms, scheme='PT', N=40)
print(f'# E1 refs done {time.time()-T0:.0f}s', flush=True)
# E2 maps: d=1 kappa x sigma fine grid; d=5 coarse
for kappa in [1.0, 2.0, 4.0]:
    ms1 = [M(1, 0.05 * k, kappa) for k in range(2, 21)]
    ref('qe', ms1, nPaths=4_000_000); ref('disc', ms1, scheme='PT', N=40); ref('disc', ms1, scheme='FT', N=40)
    ms5 = [M(5, 0.1 * k, kappa) for k in range(1, 11)]
    ref('qe', ms5); ref('disc', ms5, scheme='PT', N=40); ref('disc', ms5, scheme='FT', N=40)
print(f'# E2 refs done {time.time()-T0:.0f}s', flush=True)
# E3: N and scheme
for d in [1, 5]:
    ms = [M(d, s) for s in [0.3, 0.5, 0.7]]
    for sc in ['PT', 'FT']:
        for N in [10, 20, 40, 80, 160]:
            ref('disc', ms, scheme=sc, N=N)
# E4 extra refs (PT80 at d=2,10)
for d in [2, 10]:
    ref('disc', [M(d, 0.3), M(d, 0.7)], scheme='PT', N=80)
# E5: correlated assets: fine FT400 reference + PT40
for d in [1, 2, 5, 10, 20]:
    ms = [M(d, 0.3, rhoS=0.5), M(d, 0.7, rhoS=0.5)] if d > 1 else [M(1, 0.3), M(1, 0.7)]
    ref('disc', ms, scheme='FT', N=400, nPaths=2_000_000, seed=104)
    if d > 1: ref('disc', ms, scheme='PT', N=40)
print(f'# all refs done {time.time()-T0:.0f}s', flush=True)

# ---------------- Deep BSDE ----------------
# E1 main grid
for d in [1, 2, 5, 10, 20]:
    nS = 5 if d <= 5 else 3
    ms, ss = [], []
    for s in SIGS:
        for seed in range(1, nS + 1):
            ms.append(M(d, s)); ss.append(seed)
    bsde('E1', ms, ss, {})
# E2 Feller shift
ms, ss = [], []
for kappa in [1.0, 4.0]:
    for s in [0.2, 0.4, 0.6, 0.8, 1.0]:
        for seed in [1, 2, 3]:
            ms.append(M(1, s, kappa)); ss.append(seed)
bsde('E2', ms, ss, {})
# E3 N and scheme
for d in [1, 5]:
    combos = [('PT', 10), ('PT', 20), ('PT', 80), ('FT', 10), ('FT', 20), ('FT', 40), ('FT', 80)] if d == 1 else [('PT', 80), ('FT', 40), ('FT', 80)]
    for sc, N in combos:
        ms, ss = [], []
        for s in [0.3, 0.5, 0.7]:
            for seed in [1, 2, 3]:
                ms.append(M(d, s)); ss.append(seed)
        bsde('E3', ms, ss, {'scheme': sc, 'N': N})
# E4 ablations
for d in [2, 10]:
    variants = [{'variant': 'wider', 'width': 4 * (d + 10)}, {'variant': 'deeper', 'depth': 3}, {'variant': 'moreIters', 'iters': 10000},
                {'variant': 'largerBatch', 'batch': 1024}, {'variant': 'moreSteps', 'N': 80}, {'variant': 'warmStart', 'warmStart': True},
                {'variant': 'FTscheme', 'scheme': 'FT'}]
    for v in variants:
        ms, ss = [], []
        for s in [0.3, 0.7]:
            for seed in [1, 2, 3]:
                ms.append(M(d, s)); ss.append(seed)
        bsde('E4', ms, ss, v)
# E5 correlated assets
for d in [2, 5, 10, 20]:
    ms, ss = [], []
    for s in [0.3, 0.7]:
        for seed in [1, 2, 3]:
            ms.append(M(d, s, rhoS=0.5)); ss.append(seed)
    bsde('E5', ms, ss, {})
# Timing: single runs
for d in [1, 2, 5, 10, 20]:
    bsde('TIME', [M(d, 0.4)], [1], {})
print(f'# ALL DONE {time.time()-T0:.0f}s', flush=True)
OUT.close()
