import time
OUT = open('results_gpu.jsonl', 'a')
T0 = time.time()
SIGS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for d in [2, 5, 10, 20]:
    ms = [M(d, s) for s in SIGS]
    n = 32_000_000 if d <= 2 else (16_000_000 if d <= 5 else 8_000_000)
    ref('qe', ms, nPaths=n, seed=2001)
    ref('disc', ms, scheme='PT', N=40, nPaths=n, seed=2002)
ref('disc', [M(1, s) for s in SIGS], scheme='PT', N=40, nPaths=32_000_000, seed=2003)
for d in [1, 2, 5, 10, 20]:
    ms = [M(d, 0.3, rhoS=0.5), M(d, 0.7, rhoS=0.5)] if d > 1 else [M(1, 0.3), M(1, 0.7)]
    ref('disc', ms, scheme='FT', N=400, nPaths=16_000_000 if d <= 5 else 8_000_000, seed=2004)
    if d > 1: ref('disc', ms, scheme='PT', N=40, nPaths=16_000_000, seed=2005)
print(f'# BOOST DONE {time.time()-T0:.0f}s', flush=True)
OUT.close()
