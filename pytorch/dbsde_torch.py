# Deep BSDE under multi-asset Heston: PyTorch implementation (GPU), stacked independent runs.
# Port of the validated JavaScript reference implementation (lib.js).
import math, json, time
import torch

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
F32 = torch.float32
F64 = torch.float64
BASE = dict(S0=100.0, K=100.0, r=0.05, T=0.5, kappa=2.0, theta=0.04, rho=-0.7, v0=0.04, rhoS=0.0)


def col(vals, dtype=F32):
    return torch.tensor(vals, dtype=dtype, device=dev).view(-1, 1, 1)


def draw(gen, R, M, d, dt, rho, rhoS, dtype=F32):
    sq = math.sqrt(dt)
    eta = torch.randn(R, M, d, generator=gen, device=dev, dtype=dtype)
    if rhoS > 0:
        eta0 = torch.randn(R, M, 1, generator=gen, device=dev, dtype=dtype)
        w = sq * (math.sqrt(rhoS) * eta0 + math.sqrt(1 - rhoS) * eta)
    else:
        w = sq * eta
    b = rho * w + math.sqrt(1 - rho * rho) * sq * torch.randn(R, M, d, generator=gen, device=dev, dtype=dtype)
    return w, b


def step(scheme, S, v, dW, dB, dt, r, kappa, theta, sigma):
    vp = torch.clamp(v, min=0.0)
    sv = torch.sqrt(vp)
    if scheme == 'PT':
        S_new = S + r * S * dt + sv * S * dW
        v_new = v + kappa * (theta - v) * dt + sigma * sv * dB
    else:  # FT: full truncation + log-Euler
        S_new = S * torch.exp((r - 0.5 * vp) * dt + sv * dW)
        v_new = v + kappa * (theta - vp) * dt + sigma * sv * dB
    return S_new, v_new


def _accumulate(acc, S, d, K, r, T, S0, EB):
    """acc: dict of float64 tensors [R]; S: [R,M,d] terminal prices."""
    S = S.to(F64)
    disc = math.exp(-r * T)
    B = S.mean(dim=2)                     # [R,M]
    g = disc * torch.clamp(B - K, min=0)
    h = B - EB
    dl = torch.where(B > K, disc * S.sum(dim=2) / (d * d * S0), torch.zeros_like(B))
    for k, x in (('g', g), ('gg', g * g), ('h', h), ('hh', h * h), ('gh', g * h), ('dl', dl), ('dl2', dl * dl)):
        acc[k] = acc.get(k, 0) + x.sum(dim=1)
    acc['n'] = acc.get('n', 0) + S.shape[1]


def _finish(acc):
    n = acc['n']
    mg, mh = acc['g'] / n, acc['h'] / n
    vg, vh = acc['gg'] / n - mg ** 2, acc['hh'] / n - mh ** 2
    cgh = acc['gh'] / n - mg * mh
    beta = cgh / vh
    price = mg - beta * mh
    varcv = vg - cgh ** 2 / vh
    md = acc['dl'] / n
    out = []
    for i in range(len(mg)):
        out.append(dict(price=price[i].item(), se=math.sqrt(max(varcv[i].item(), 0) / n), plainPrice=mg[i].item(),
                        plainSe=math.sqrt(max(vg[i].item(), 0) / n), payoffStd=math.sqrt(max(vg[i].item(), 0)),
                        delta=md[i].item(), deltaSe=math.sqrt(max((acc['dl2'][i] / n - md[i] ** 2).item(), 0) / n), nPaths=n))
    return out


def mc_disc(models, scheme, N, nPaths, seed, chunk=None):
    """Stacked MC of discretised dynamics. models: list of dicts sharing d, rhoS."""
    d, rhoS = models[0]['d'], models[0].get('rhoS', 0.0)
    p = {**BASE, **models[0]}
    R = len(models)
    chunk = chunk or max(1024, int(2_000_000 // (R * d)))
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    dt = p['T'] / N
    sigma = col([m['sigma'] for m in models]); kappa = col([m.get('kappa', BASE['kappa']) for m in models])
    EB = p['S0'] * ((1 + p['r'] * dt) ** N if scheme == 'PT' else math.exp(p['r'] * p['T']))
    acc = {}; done = 0
    while done < nPaths:
        M = min(chunk, nPaths - done)
        S = torch.full((R, M, d), p['S0'], device=dev, dtype=F32)
        v = torch.full((R, M, d), p['v0'], device=dev, dtype=F32)
        for _ in range(N):
            dW, dB = draw(gen, R, M, d, dt, p['rho'], rhoS)
            S, v = step(scheme, S, v, dW, dB, dt, p['r'], kappa, p['theta'], sigma)
        _accumulate(acc, S, d, p['K'], p['r'], p['T'], p['S0'], EB)
        done += M
    return _finish(acc)


def mc_qe(models, N, nPaths, seed, chunk=None):
    """Andersen (2008) QE with martingale correction, gamma1=gamma2=1/2. rhoS must be 0."""
    d = models[0]['d']; p = {**BASE, **models[0]}
    assert not p.get('rhoS'), 'QE needs rhoS=0'
    R = len(models)
    chunk = chunk or max(1024, int(2_000_000 // (R * d)))
    gen = torch.Generator(device=dev); gen.manual_seed(seed)
    dt = p['T'] / N; r, theta, rho = p['r'], p['theta'], p['rho']
    sig = col([m['sigma'] for m in models], F64); kap = col([m.get('kappa', BASE['kappa']) for m in models], F64)
    E = torch.exp(-kap * dt); c1 = sig ** 2 * E * (1 - E) / kap; c2 = theta * sig ** 2 * (1 - E) ** 2 / (2 * kap)
    K1 = 0.5 * dt * (kap * rho / sig - 0.5) - rho / sig; K2 = 0.5 * dt * (kap * rho / sig - 0.5) + rho / sig
    K3 = 0.5 * dt * (1 - rho * rho); K4 = K3; A = K2 + 0.5 * K4
    E, c1, c2, K1, K2, A = [x.to(F32) for x in (E, c1, c2, K1, K2, A)]
    EB = p['S0'] * math.exp(r * p['T'])
    acc = {}; done = 0
    while done < nPaths:
        M = min(chunk, nPaths - done)
        lnS = torch.full((R, M, d), math.log(p['S0']), device=dev, dtype=F32)
        v = torch.full((R, M, d), p['v0'], device=dev, dtype=F32)
        for _ in range(N):
            m = theta + (v - theta) * E
            s2 = v * c1 + c2
            psi = s2 / (m * m)
            Z = torch.randn(R, M, d, generator=gen, device=dev, dtype=F32)
            U = torch.rand(R, M, d, generator=gen, device=dev, dtype=F32)
            Zs = torch.randn(R, M, d, generator=gen, device=dev, dtype=F32)
            ip = 2.0 / psi
            b2 = ip - 1 + torch.sqrt(ip) * torch.sqrt(torch.clamp(ip - 1, min=0))
            a = m / (1 + b2)
            vq = a * (torch.sqrt(torch.clamp(b2, min=0)) + Z) ** 2
            den = torch.clamp(1 - 2 * A * a, min=1e-12)
            Mq = torch.exp(A * b2 * a / den) / torch.sqrt(den)
            pp = (psi - 1) / (psi + 1)
            beta = (1 - pp) / m
            ve = torch.where(U <= pp, torch.zeros_like(U), torch.log(torch.clamp((1 - pp) / (1 - U), min=1e-30)) / beta)
            Me = pp + beta * (1 - pp) / (beta - A)
            quad = psi <= 1.5
            vn = torch.where(quad, vq, ve)
            Mg = torch.where(quad, Mq, Me)
            K0s = -torch.log(Mg) - (K1 + 0.5 * K3) * v
            lnS = lnS + r * dt + K0s + K1 * v + K2 * vn + torch.sqrt(torch.clamp(K3 * v + K4 * vn, min=0)) * Zs
            v = vn
        _accumulate(acc, torch.exp(lnS), d, p['K'], r, p['T'], p['S0'], EB)
        done += M
    return _finish(acc)


def train_stack(models, seeds, cfg):
    """Train R independent Deep BSDE solvers in parallel (one per (model, seed)).
    All models share d, rhoS; cfg shared: N, batch, iters, width, depth, scheme, warmStart, lr..."""
    R = len(models); d = models[0]['d']; rhoS = models[0].get('rhoS', 0.0); p = {**BASE, **models[0]}
    D2 = 2 * d
    N = cfg.get('N', 40); M = cfg.get('batch', 256); iters = cfg.get('iters', 4000); depth = cfg.get('depth', 2)
    h = cfg.get('width') or (d + 10); scheme = cfg.get('scheme', 'PT')
    dt = p['T'] / N; S0, K, r, theta, v0, rho = p['S0'], p['K'], p['r'], p['theta'], p['v0'], p['rho']
    gen = torch.Generator(device=dev); gen.manual_seed(int(1000003 * seeds[0] + 7919 * R + 13 * d + N))
    sigma = col([m['sigma'] for m in models]); kappa = col([m.get('kappa', BASE['kappa']) for m in models])
    zs = S0 * math.sqrt(theta); Df = 1 + r * dt; sTh = math.sqrt(theta)
    # params
    dims = [D2] + [h] * depth + [D2]
    Ws, bs = [], []
    for l in range(depth + 1):
        sd = math.sqrt(2.0 / dims[l]) * (0.1 if l == depth else 1.0)
        Ws.append((torch.randn(R, N - 1, dims[l], dims[l + 1], generator=gen, device=dev) * sd).requires_grad_())
        bs.append(torch.zeros(R, N - 1, 1, dims[l + 1], device=dev, requires_grad=True))
    if cfg.get('warmStart'):
        y0init = [mc_disc([m], scheme, N, 5000, 999 + s)[0]['plainPrice'] for m, s in zip(models, seeds)]
        Y0 = torch.tensor(y0init, device=dev, dtype=F32)
    else:
        fwd = max(S0 * math.exp(r * p['T']) - K, 1.0)
        Y0 = (0.5 + torch.rand(R, generator=gen, device=dev)) * fwd * 2
    Y0 = Y0.clone().requires_grad_()
    Z0 = torch.zeros(R, D2, device=dev, requires_grad=True)
    base_lr = cfg.get('lr', 3e-3)
    opt = torch.optim.Adam([{'params': [Y0], 'lr': cfg.get('y0lr', 0.05)}, {'params': [Z0], 'lr': cfg.get('z0lr', 0.01)},
                            {'params': Ws + bs, 'lr': base_lr}], betas=(0.9, 0.999), eps=1e-8)
    init_lrs = [g['lr'] for g in opt.param_groups]
    sched = cfg.get('schedule', [(0.5, 1.0), (0.8, 0.1), (1.0, 0.01)])
    powN = torch.tensor([Df ** (N - 1 - n) for n in range(N)], device=dev, dtype=F32).view(1, N, 1)

    def simulate(Mb):
        S = torch.full((R, Mb, d), S0, device=dev); v = torch.full((R, Mb, d), v0, device=dev)
        Xs, dWs = [], []
        for n in range(N):
            Xs.append(torch.cat([(S / S0 - 1) / sTh, (v - theta) / theta], dim=2))
            dW, dB = draw(gen, R, Mb, d, dt, rho, rhoS)
            dWs.append(torch.cat([dW, dB], dim=2))
            S, v = step(scheme, S, v, dW, dB, dt, r, kappa, theta, sigma)
        g = torch.clamp(S.mean(dim=2) - K, min=0)  # [R,Mb]
        return torch.stack(Xs, 1), torch.stack(dWs, 1), g   # [R,N,Mb,2d]

    def rollout(X, dWb, Mb):
        H = X[:, 1:].reshape(R * (N - 1), Mb, D2)
        for l in range(depth + 1):
            H = torch.bmm(H, Ws[l].reshape(R * (N - 1), dims[l], dims[l + 1])) + bs[l].reshape(R * (N - 1), 1, dims[l + 1])
            if l < depth:
                H = torch.relu(H)
        Z = zs * H.reshape(R, N - 1, Mb, D2)
        incr = torch.cat([(zs * Z0).view(R, 1, 1, D2).expand(R, 1, Mb, D2) * dWb[:, :1], Z * dWb[:, 1:]], dim=1).sum(-1)  # [R,N,Mb]
        YN = Y0.view(R, 1) * Df ** N + (incr * powN).sum(1)
        return YN

    t0 = time.time(); hist = []
    for it in range(iters):
        frac = it / iters
        f = next(v for b, v in sched if frac < b)
        for g_, lr0 in zip(opt.param_groups, init_lrs):
            g_['lr'] = lr0 * f
        X, dWb, g = simulate(M)
        YN = rollout(X, dWb, M)
        lossR = ((YN - g) ** 2).mean(1)
        opt.zero_grad(set_to_none=True)
        lossR.sum().backward()
        opt.step()
        if it % 100 == 0 or it == iters - 1:
            hist.append([it, Y0.detach().cpu().tolist(), lossR.detach().cpu().tolist()])
    if dev == 'cuda':
        torch.cuda.synchronize()
    train_s = time.time() - t0
    # validation
    nVal = cfg.get('nVal', 8192); sse = torch.zeros(R, device=dev, dtype=F64); sg = torch.zeros_like(sse); sgg = torch.zeros_like(sse); cnt = 0
    with torch.no_grad():
        while cnt < nVal:
            Mb = 1024
            X, dWb, g = simulate(Mb); YN = rollout(X, dWb, Mb)
            sse += ((YN - g) ** 2).to(F64).sum(1); sg += g.to(F64).sum(1); sgg += (g.to(F64) ** 2).sum(1); cnt += Mb
    mse = (sse / cnt); gm = sg / cnt; gvar = sgg / cnt - gm ** 2
    out = []
    z0 = (zs * Z0).detach().cpu()
    for i in range(R):
        dvec = (z0[i, :d] / (math.sqrt(v0) * S0)).tolist()
        out.append(dict(price=Y0[i].item(), delta=sum(dvec) / d, z0=z0[i].tolist(), valMSE=mse[i].item(), payoffVar=gvar[i].item(),
                        hedgeVarReduction=1 - mse[i].item() / gvar[i].item(),
                        finalTrainLoss=sum(hh[2][i] for hh in hist[-5:]) / 5,
                        trainSecStack=train_s, R=R,
                        nParams=1 + D2 + (N - 1) * sum(dims[l] * dims[l + 1] + dims[l + 1] for l in range(depth + 1)),
                        hist=[[hh[0], round(hh[1][i], 5), round(hh[2][i], 4)] for hh in hist[::5]]))
    return out
