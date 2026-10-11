'use strict';
// Deep BSDE under multi-asset Heston: core library (pure JS, no deps).
// Contents: seeded RNG, Heston analytic (d=1), reference Monte Carlo (QE-M and Euler schemes),
// Deep BSDE solver with manual backprop + Adam.

// ---------------- RNG ----------------
class RNG {
  constructor(seed) {
    // splitmix32 to init sfc32 state
    let s = (seed >>> 0) ^ 0x9e3779b9;
    const sm = () => { s = (s + 0x9e3779b9) | 0; let z = s; z = Math.imul(z ^ (z >>> 16), 0x85ebca6b); z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35); return (z ^ (z >>> 16)) >>> 0; };
    this.a = sm(); this.b = sm(); this.c = sm(); this.d = sm();
    for (let i = 0; i < 20; i++) this.u32();
    this.hasSpare = false; this.spare = 0;
  }
  u32() {
    let a = this.a, b = this.b, c = this.c, d = this.d;
    const t = (((a + b) | 0) + d) | 0;
    d = (d + 1) | 0; a = b ^ (b >>> 9); b = (c + (c << 3)) | 0; c = (c << 21) | (c >>> 11); c = (c + t) | 0;
    this.a = a; this.b = b; this.c = c; this.d = d;
    return t >>> 0;
  }
  uniform() { return (this.u32() + 0.5) / 4294967296; } // in (0,1)
  normal() {
    if (this.hasSpare) { this.hasSpare = false; return this.spare; }
    let u, v, s;
    do { u = 2 * this.uniform() - 1; v = 2 * this.uniform() - 1; s = u * u + v * v; } while (s >= 1 || s === 0);
    const f = Math.sqrt(-2 * Math.log(s) / s);
    this.spare = v * f; this.hasSpare = true; return u * f;
  }
}

// ---------------- Complex helpers (as [re, im]) ----------------
const C = {
  add: (a, b) => [a[0] + b[0], a[1] + b[1]],
  sub: (a, b) => [a[0] - b[0], a[1] - b[1]],
  mul: (a, b) => [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]],
  div: (a, b) => { const den = b[0] * b[0] + b[1] * b[1]; return [(a[0] * b[0] + a[1] * b[1]) / den, (a[1] * b[0] - a[0] * b[1]) / den]; },
  exp: (a) => { const e = Math.exp(a[0]); return [e * Math.cos(a[1]), e * Math.sin(a[1])]; },
  log: (a) => [Math.log(Math.hypot(a[0], a[1])), Math.atan2(a[1], a[0])],
  sqrt: (a) => { const r = Math.hypot(a[0], a[1]); const re = Math.sqrt((r + a[0]) / 2); const im = Math.sign(a[1] || 1) * Math.sqrt(Math.max(0, (r - a[0]) / 2)); return [re, im]; },
  sc: (s, a) => [s * a[0], s * a[1]],
};

// Characteristic function of ln S_T under Heston ("little trap" form, Albrecher et al. 2007), complex argument w
function hestonCF(w, p) {
  const { S0, r, T, kappa, theta, sigma, rho, v0 } = p;
  const iw = [-w[1], w[0]];                           // i*w
  const w2 = C.mul(w, w);
  const a = C.sub([kappa, 0], C.sc(rho * sigma, iw));  // kappa - i rho sigma w
  const D = C.sqrt(C.add(C.mul(a, a), C.sc(sigma * sigma, C.add(iw, w2))));
  const G = C.div(C.sub(a, D), C.add(a, D));
  const eDT = C.exp(C.sc(-T, D));
  const one = [1, 0];
  const num = C.sub(one, C.mul(G, eDT)), den = C.sub(one, G);
  const term1 = C.sc(Math.log(S0) + r * T, iw);
  const term2 = C.sc(kappa * theta / (sigma * sigma), C.sub(C.sc(T, C.sub(a, D)), C.sc(2, C.log(C.div(num, den)))));
  const term3 = C.sc(v0 / (sigma * sigma), C.div(C.mul(C.sub(a, D), C.sub(one, eDT)), num));
  return C.exp(C.add(term1, C.add(term2, term3)));
}

// European call price and delta (d=1). Simpson integration on (0, U].
function hestonCall(p, K, U = 400, n = 40000) {
  const lnK = Math.log(K);
  const phiMi = hestonCF([0, -1], p); // = S0 e^{rT}
  const h = U / n;
  let s1 = 0, s2 = 0;
  for (let k = 0; k <= n; k++) {
    const u = k === 0 ? 1e-10 : k * h;
    const wgt = (k === 0 || k === n) ? 1 : (k % 2 ? 4 : 2);
    const e = C.exp([0, -u * lnK]);
    const iu = [0, u];
    const f1 = C.div(C.mul(e, hestonCF([u, -1], p)), C.mul(iu, phiMi));
    const f2 = C.div(C.mul(e, hestonCF([u, 0], p)), iu);
    s1 += wgt * f1[0]; s2 += wgt * f2[0];
  }
  const P1 = 0.5 + (s1 * h / 3) / Math.PI, P2 = 0.5 + (s2 * h / 3) / Math.PI;
  return { price: p.S0 * P1 - K * Math.exp(-p.r * p.T) * P2, delta: P1 };
}

function bsCall(S, K, T, r, sig) {
  const N = (x) => 0.5 * (1 + erf(x / Math.SQRT2));
  const d1 = (Math.log(S / K) + (r + sig * sig / 2) * T) / (sig * Math.sqrt(T)), d2 = d1 - sig * Math.sqrt(T);
  return S * N(d1) - K * Math.exp(-r * T) * N(d2);
}
function erf(x) { // Abramowitz-Stegun 7.1.26 is too crude; use high-precision series/continued fraction
  const t = 1 / (1 + 0.5 * Math.abs(x));
  const y = 1 - t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? y : -y;
}

// ---------------- Path simulation ----------------
// Model params: {d, S0, K, r, T, kappa, theta, sigma (vol-of-vol), rho (S-v corr), v0, rhoS (inter-asset corr of W's)}
// Brownian structure: W_i = sqrt(rhoS)*eta0 + sqrt(1-rhoS)*eta_i ; B_i = rho W_i + sqrt(1-rho^2) Wt_i (Wt_i iid)

// One step of the discretised forward SDE for a batch, in place.
// scheme: 'PT' = original paper (Euler on S, partial truncation: raw v in drift, v+ in diffusion)
//         'FT' = full truncation (Lord et al. 2010) for v + log-Euler for S
function stepForward(scheme, p, S, v, dW, dB, dt, M) {
  const d = p.d;
  for (let m = 0; m < M; m++) {
    for (let i = 0; i < d; i++) {
      const k = m * d + i;
      const vk = v[k], vp = vk > 0 ? vk : 0, sv = Math.sqrt(vp);
      if (scheme === 'PT') {
        S[k] = S[k] + p.r * S[k] * dt + sv * S[k] * dW[k];
        v[k] = vk + p.kappa * (p.theta - vk) * dt + p.sigma * sv * dB[k];
      } else {
        S[k] = S[k] * Math.exp((p.r - 0.5 * vp) * dt + sv * dW[k]);
        v[k] = vk + p.kappa * (p.theta - vp) * dt + p.sigma * sv * dB[k];
      }
    }
  }
}

function drawIncrements(rng, p, dW, dB, dt, M) {
  const d = p.d, sq = Math.sqrt(dt), a = Math.sqrt(p.rhoS || 0), b = Math.sqrt(1 - (p.rhoS || 0)), c = Math.sqrt(1 - p.rho * p.rho);
  for (let m = 0; m < M; m++) {
    const eta0 = a > 0 ? rng.normal() : 0;
    for (let i = 0; i < d; i++) {
      const k = m * d + i;
      const w = sq * (a * eta0 + b * rng.normal());
      dW[k] = w; dB[k] = p.rho * w + c * sq * rng.normal();
    }
  }
}

// Monte Carlo for the *discretised* process (same scheme as BSDE forward), with basket control variate
// and pathwise delta. Returns price, se, delta, deltaSe.
function mcDiscrete(p, scheme, N, nPaths, seed, chunk = 4096) {
  const rng = new RNG(seed), d = p.d, dt = p.T / N;
  const S = new Float64Array(chunk * d), v = new Float64Array(chunk * d), dW = new Float64Array(chunk * d), dB = new Float64Array(chunk * d);
  const EB = scheme === 'PT' ? p.S0 * Math.pow(1 + p.r * dt, N) : p.S0 * Math.exp(p.r * p.T);
  return mcLoop(p, nPaths, chunk, EB, (M) => {
    S.fill(p.S0); v.fill(p.v0);
    for (let n = 0; n < N; n++) { drawIncrements(rng, p, dW, dB, dt, M); stepForward(scheme, p, S, v, dW, dB, dt, M); }
    return S;
  });
}

// Andersen (2008) QE scheme with martingale correction (QE-M), gamma1=gamma2=1/2. Only for rhoS=0.
function mcQE(p, N, nPaths, seed, chunk = 4096) {
  if (p.rhoS) throw new Error('QE reference requires rhoS=0');
  const rng = new RNG(seed), d = p.d, dt = p.T / N;
  const { kappa, theta, sigma, rho, r } = p;
  const E = Math.exp(-kappa * dt), c1 = sigma * sigma * E * (1 - E) / kappa, c2 = theta * sigma * sigma * (1 - E) * (1 - E) / (2 * kappa);
  const K1 = 0.5 * dt * (kappa * rho / sigma - 0.5) - rho / sigma, K2 = 0.5 * dt * (kappa * rho / sigma - 0.5) + rho / sigma;
  const K3 = 0.5 * dt * (1 - rho * rho), K4 = K3, A = K2 + 0.5 * K4;
  const lnS = new Float64Array(chunk * d), v = new Float64Array(chunk * d), S = new Float64Array(chunk * d);
  const EB = p.S0 * Math.exp(r * p.T);
  return mcLoop(p, nPaths, chunk, EB, (M) => {
    lnS.fill(Math.log(p.S0)); v.fill(p.v0);
    for (let n = 0; n < N; n++) {
      for (let k = 0; k < M * d; k++) {
        const vk = v[k];
        const m = theta + (vk - theta) * E, s2 = vk * c1 + c2, psi = s2 / (m * m);
        let vn, Mg;
        if (psi <= 1.5) {
          const ip = 2 / psi, b2 = ip - 1 + Math.sqrt(ip) * Math.sqrt(ip - 1), a = m / (1 + b2), b = Math.sqrt(b2);
          const z = rng.normal(); vn = a * (b + z) * (b + z);
          Mg = Math.exp(A * b2 * a / (1 - 2 * A * a)) / Math.sqrt(1 - 2 * A * a);
        } else {
          const pp = (psi - 1) / (psi + 1), beta = (1 - pp) / m, u = rng.uniform();
          vn = u <= pp ? 0 : Math.log((1 - pp) / (1 - u)) / beta;
          Mg = pp + beta * (1 - pp) / (beta - A);
        }
        const K0s = -Math.log(Mg) - (K1 + 0.5 * K3) * vk;
        lnS[k] += r * dt + K0s + K1 * vk + K2 * vn + Math.sqrt(K3 * vk + K4 * vn) * rng.normal();
        v[k] = vn;
      }
    }
    for (let k = 0; k < M * d; k++) S[k] = Math.exp(lnS[k]);
    return S;
  });
}

// Shared MC accumulation: discounted basket call with control variate (basket mean, known expectation EB),
// pathwise delta (dPrice/dS_i(0), averaged over i by symmetry).
function mcLoop(p, nPaths, chunk, EB, simulate) {
  const d = p.d, disc = Math.exp(-p.r * p.T);
  let n = 0, sg = 0, sgg = 0, sh = 0, shh = 0, sgh = 0, sdl = 0, sdl2 = 0;
  while (n < nPaths) {
    const M = Math.min(chunk, nPaths - n);
    const S = simulate(M);
    for (let m = 0; m < M; m++) {
      let B = 0; for (let i = 0; i < d; i++) B += S[m * d + i]; B /= d;
      const g = disc * Math.max(B - p.K, 0), h = B - EB;
      let dl = 0; if (B > p.K) { for (let i = 0; i < d; i++) dl += S[m * d + i]; dl = disc * dl / (d * d * p.S0); }
      sg += g; sgg += g * g; sh += h; shh += h * h; sgh += g * h; sdl += dl; sdl2 += dl * dl;
    }
    n += M;
  }
  const mg = sg / n, mh = sh / n, vg = sgg / n - mg * mg, vh = shh / n - mh * mh, cgh = sgh / n - mg * mh;
  const beta = cgh / vh, price = mg - beta * mh, varCV = vg - cgh * cgh / vh;
  const md = sdl / n;
  return { price, se: Math.sqrt(varCV / n), plainPrice: mg, plainSe: Math.sqrt(vg / n), payoffStd: Math.sqrt(vg), delta: md, deltaSe: Math.sqrt((sdl2 / n - md * md) / n), nPaths: n };
}

// ---------------- Deep BSDE ----------------
// Y_{n+1} = Y_n (1 + r dt) + Z_n . dW_n ,  dW_n = (dW_1..dW_d, dB_1..dB_d)
// Z_0: trainable vector (X_0 deterministic). Z_n, n>=1: MLP_n(normalised X_n), 2d -> h -> h (-> h) -> 2d, ReLU.
function trainDeepBSDE(cfg) {
  const p = cfg.model, d = p.d, D2 = 2 * d;
  const N = cfg.N || 40, M = cfg.batch || 256, iters = cfg.iters || 4000, depth = cfg.depth || 2;
  const h = cfg.width || (d + 10), scheme = cfg.scheme || 'PT', dt = p.T / N;
  const rng = new RNG(cfg.seed * 7919 + 17);
  const zs = p.S0 * Math.sqrt(p.theta);  // output scale for Z
  const Df = 1 + p.r * dt;
  // ---- parameter layout ----
  const layers = []; // per step n=1..N-1 : list of {inD,outD,wOff,bOff}
  let off = 0; const y0Off = off; off += 1; const z0Off = off; off += D2;
  for (let n = 1; n < N; n++) {
    const L = []; let inD = D2;
    for (let l = 0; l <= depth; l++) { const outD = l < depth ? h : D2; L.push({ inD, outD, wOff: off, bOff: off + inD * outD }); off += inD * outD + outD; inD = outD; }
    layers.push(L);
  }
  const P = new Float64Array(off), G = new Float64Array(off), Am = new Float64Array(off), Av = new Float64Array(off);
  const lrScale = new Float64Array(off).fill(1);
  // init
  for (const L of layers) L.forEach((ly, li) => {
    const sd = Math.sqrt(2 / ly.inD) * (li === L.length - 1 ? 0.1 : 1);
    for (let k = 0; k < ly.inD * ly.outD; k++) P[ly.wOff + k] = sd * rng.normal();
  });
  // Y0 init: either warm start (MC with 5000 paths, as in original paper) or a crude uniform guess
  if (cfg.warmStart) P[y0Off] = mcDiscrete(p, scheme, N, 5000, cfg.seed + 999).plainPrice;
  else P[y0Off] = (0.5 + rng.uniform()) * Math.max(p.S0 * Math.exp(p.r * p.T) - p.K, 1) * 2; // uninformative
  for (let k = 0; k < D2; k++) P[z0Off + k] = 0;
  const baseLR = cfg.lr || 3e-3, y0LR = cfg.y0lr || 0.05, z0LR = cfg.z0lr || 0.01;
  lrScale[y0Off] = y0LR / baseLR; for (let k = 0; k < D2; k++) lrScale[z0Off + k] = z0LR / baseLR;
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  // ---- buffers ----
  const Sb = new Float64Array(M * d), vb = new Float64Array(M * d), dWb = new Float64Array(M * d), dBb = new Float64Array(M * d);
  const X = Array.from({ length: N }, () => new Float64Array(M * D2));     // normalised inputs per step
  const dWall = Array.from({ length: N }, () => new Float64Array(M * D2)); // increments per step
  const acts = layers.map(L => L.map(ly => new Float64Array(M * ly.outD))); // activations (post-ReLU for hidden, raw for output)
  const Y = new Float64Array(M), e = new Float64Array(M), g = new Float64Array(M);
  const gradBuf = new Float64Array(M * Math.max(h, D2)), gradBuf2 = new Float64Array(M * Math.max(h, D2));
  const sTh = Math.sqrt(p.theta);

  function simulate(Mb) {
    Sb.fill(p.S0); vb.fill(p.v0);
    for (let n = 0; n < N; n++) {
      // inputs at step n (before stepping)
      const Xn = X[n];
      for (let m = 0; m < Mb; m++) for (let i = 0; i < d; i++) {
        Xn[m * D2 + i] = (Sb[m * d + i] / p.S0 - 1) / sTh;
        Xn[m * D2 + d + i] = (vb[m * d + i] - p.theta) / p.theta;
      }
      drawIncrements(rng, p, dWb, dBb, dt, Mb);
      const Wn = dWall[n];
      for (let m = 0; m < Mb; m++) for (let i = 0; i < d; i++) { Wn[m * D2 + i] = dWb[m * d + i]; Wn[m * D2 + d + i] = dBb[m * d + i]; }
      stepForward(scheme, p, Sb, vb, dWb, dBb, dt, Mb);
    }
    const disc = 1; // payoff at T, undiscounted
    for (let m = 0; m < Mb; m++) { let B = 0; for (let i = 0; i < d; i++) B += Sb[m * d + i]; g[m] = disc * Math.max(B / d - p.K, 0); }
  }
  function forwardNet(n, Mb) { // computes Z for step n (n>=1), returns output buffer (unscaled)
    const L = layers[n - 1], A = acts[n - 1];
    let inp = X[n];
    for (let l = 0; l < L.length; l++) {
      const ly = L[l], out = A[l], W = ly.wOff, bo = ly.bOff, inD = ly.inD, outD = ly.outD, relu = l < L.length - 1;
      for (let m = 0; m < Mb; m++) {
        const io = m * inD, oo = m * outD;
        for (let j = 0; j < outD; j++) out[oo + j] = P[bo + j];
        for (let q = 0; q < inD; q++) { const a = inp[io + q]; if (a === 0) continue; const wr = W + q * outD; for (let j = 0; j < outD; j++) out[oo + j] += a * P[wr + j]; }
        if (relu) for (let j = 0; j < outD; j++) if (out[oo + j] < 0) out[oo + j] = 0;
      }
      inp = out;
    }
    return A[L.length - 1];
  }
  function rollout(Mb) { // forward pass of BSDE, fills Y (terminal)
    for (let m = 0; m < Mb; m++) { let z = 0; const Wn = dWall[0]; for (let j = 0; j < D2; j++) z += zs * P[z0Off + j] * Wn[m * D2 + j]; Y[m] = P[y0Off] * Df + z; }
    for (let n = 1; n < N; n++) {
      const O = forwardNet(n, Mb), Wn = dWall[n];
      for (let m = 0; m < Mb; m++) { let z = 0; for (let j = 0; j < D2; j++) z += zs * O[m * D2 + j] * Wn[m * D2 + j]; Y[m] = Y[m] * Df + z; }
    }
  }
  function backward(Mb) {
    G.fill(0);
    let loss = 0;
    for (let m = 0; m < Mb; m++) { e[m] = Y[m] - g[m]; loss += e[m] * e[m]; }
    loss /= Mb;
    const DN = Math.pow(Df, N);
    for (let m = 0; m < Mb; m++) G[y0Off] += 2 / Mb * e[m] * DN;
    { const c = Math.pow(Df, N - 1), Wn = dWall[0]; for (let m = 0; m < Mb; m++) { const cm = 2 / Mb * e[m] * c * zs; for (let j = 0; j < D2; j++) G[z0Off + j] += cm * Wn[m * D2 + j]; } }
    for (let n = 1; n < N; n++) {
      const L = layers[n - 1], A = acts[n - 1], Wn = dWall[n], c = Math.pow(Df, N - 1 - n);
      // grad wrt output
      let dOut = gradBuf; for (let m = 0; m < Mb; m++) { const cm = 2 / Mb * e[m] * c * zs; for (let j = 0; j < D2; j++) dOut[m * D2 + j] = cm * Wn[m * D2 + j]; }
      for (let l = L.length - 1; l >= 0; l--) {
        const ly = L[l], inD = ly.inD, outD = ly.outD, inp = l === 0 ? X[n] : A[l - 1];
        // dW = inp^T dOut ; db = sum dOut
        for (let m = 0; m < Mb; m++) {
          const io = m * inD, oo = m * outD;
          for (let j = 0; j < outD; j++) G[ly.bOff + j] += dOut[oo + j];
          for (let q = 0; q < inD; q++) { const a = inp[io + q]; if (a === 0) continue; const wr = ly.wOff + q * outD; for (let j = 0; j < outD; j++) G[wr + j] += a * dOut[oo + j]; }
        }
        if (l > 0) { // dIn = dOut W^T, masked by ReLU of inp (post-activation >0)
          const dIn = dOut === gradBuf ? gradBuf2 : gradBuf;
          for (let m = 0; m < Mb; m++) {
            const io = m * inD, oo = m * outD;
            for (let q = 0; q < inD; q++) {
              if (inp[io + q] <= 0) { dIn[io + q] = 0; continue; }
              const wr = ly.wOff + q * outD; let s = 0; for (let j = 0; j < outD; j++) s += P[wr + j] * dOut[oo + j]; dIn[io + q] = s;
            }
          }
          dOut = dIn;
        }
      }
    }
    return loss;
  }
  if (cfg.gradCheck) {
    simulate(M); rollout(M); backward(M); const Gc = Float64Array.from(G); const out = [];
    const lossAt = () => { rollout(M); let l = 0; for (let m = 0; m < M; m++) l += (Y[m] - g[m]) ** 2; return l / M; };
    const idx = [y0Off, z0Off, z0Off + D2 - 1]; for (let t = 0; t < 12; t++) idx.push(z0Off + D2 + Math.floor(rng.uniform() * (off - z0Off - D2)));
    for (const k of idx) { const hh = 1e-5 * Math.max(1, Math.abs(P[k])); const o = P[k]; P[k] = o + hh; const lp = lossAt(); P[k] = o - hh; const lm = lossAt(); P[k] = o; out.push({ k, analytic: Gc[k], fd: (lp - lm) / (2 * hh) }); }
    return out;
  }
  // ---- training loop ----
  const hist = []; const t0 = Date.now();
  const sched = cfg.schedule || [[0.5, 1], [0.8, 0.1], [1.0, 0.01]];
  let bc1 = 1, bc2 = 1;
  for (let it = 0; it < iters; it++) {
    const frac = it / iters; let f = 1; for (const [b, v] of sched) { if (frac < b) { f = v; break; } }
    simulate(M); rollout(M); const loss = backward(M);
    bc1 *= b1; bc2 *= b2;
    const lr = baseLR * f;
    for (let k = 0; k < off; k++) {
      const gk = G[k]; Am[k] = b1 * Am[k] + (1 - b1) * gk; Av[k] = b2 * Av[k] + (1 - b2) * gk * gk;
      P[k] -= lr * lrScale[k] * (Am[k] / (1 - bc1)) / (Math.sqrt(Av[k] / (1 - bc2)) + eps);
    }
    if (it % 100 === 0 || it === iters - 1) hist.push([it, P[y0Off], loss]);
  }
  const trainMs = Date.now() - t0;
  // ---- validation on fresh paths: hedging residual ----
  const nVal = cfg.nVal || 8192; let sse = 0, sg = 0, sgg = 0, cnt = 0;
  while (cnt < nVal) { simulate(M); rollout(M); for (let m = 0; m < M; m++) { const r = Y[m] - g[m]; sse += r * r; sg += g[m]; sgg += g[m] * g[m]; } cnt += M; }
  const mse = sse / cnt, gm = sg / cnt, gvar = sgg / cnt - gm * gm;
  const z0 = Array.from({ length: D2 }, (_, j) => zs * P[z0Off + j]);
  const deltaVec = z0.slice(0, d).map(z => z / (Math.sqrt(p.v0) * p.S0));
  return {
    price: P[y0Off], z0, delta: deltaVec.reduce((a, b) => a + b, 0) / d, deltaVec,
    valMSE: mse, payoffVar: gvar, hedgeVarReduction: 1 - mse / gvar, finalTrainLoss: hist[hist.length - 1][2],
    trainMs, nParams: off, hist,
  };
}

// Finite-difference gradient check helper (exposed for tests)
module.exports = { RNG, hestonCF, hestonCall, bsCall, mcDiscrete, mcQE, trainDeepBSDE, stepForward, drawIncrements };
