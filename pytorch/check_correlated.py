# Convergence check for the correlated-asset case (run after dbsde_torch.py and run_all.py definitions are loaded)
import time
OUT = open('results_gpu.jsonl', 'a')
for d in [5, 10]:
    for sc, N in [('PT',40),('PT',160),('PT',640),('FT',40),('FT',160),('FT',640)]:
        ref('disc', [M(d,0.7,rhoS=0.5)], scheme=sc, N=N, nPaths=4_000_000, seed=3000+N+d)
for sc, N in [('PT',640),('FT',640)]:
    ref('disc', [M(10,0.7)], scheme=sc, N=N, nPaths=4_000_000, seed=3100+N)
OUT.close()
