#!/usr/bin/env python3
"""Audit of the numerical part of K. Nath and L. Xie, "Almost primes and primes that are sums of two squares
plus 1", arXiv:2501.16723v3 (17 Aug 2026), Theorem 1.1 (Omega(p+2) <= 9 for p = m^2 + n^2 + 1).

Pure Python standard library.  Usage:  python code/verify_nath_xie.py [--quick]
Output: data/verify_nath_xie.txt (and the same text on stdout).

Sections of the output
  1. sieve functions: half-dimensional (F1, f1) and linear (F2, f2); correct system, the system printed on p. 23,
     and the system coded in Computation_files/lower_f_sigma.m (history F = 2e^g/s, f = 0 on [1,2]).
  2. constants C, kappa, c1, c2, c3: as printed / as coded / corrected (c3 with the factor of p = 2; joint
     singular series of the two linear sieves in Proposition 4.2).
  3. a direct numerical check of the Wirsing constant c3 (summation of b*(m) m / phi(m)^2 up to 10^7).
  4. the functionals G, I, H of Section 5 at the authors' points, for each set of functions.
  5. what the method gives with the correct functions: unweighted (G > 0) and weighted (H > 0), with
     R0 = constants as printed, R1 = corrected constants, R2 = R1 + free choice of the levels in Proposition 4.2.
"""
import math, os, sys, time

T0 = time.time()
QUICK = "--quick" in sys.argv
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "data")
os.makedirs(OUTDIR, exist_ok=True)
_out = []


def say(s=""):
    print(s, flush=True)
    _out.append(s)


def clock():
    return "[%5.1f s]" % (time.time() - T0)


GAMMA = 0.57721566490153286
EG = math.exp(GAMMA)
AH = 2.0 * math.sqrt(EG / math.pi)          # sqrt(s) F_{1/2}(s) = AH on (0, 2]


# ------------------------------------------------------------------------------------------------------------
# Gauss-Legendre nodes
# ------------------------------------------------------------------------------------------------------------
def gauss_legendre(n):
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-15:
                break
        xs.append(x)
        ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


GLX, GLW = gauss_legendre(40)
GL5X, GL5W = gauss_legendre(5)


def gl(fun, a, b, xs=GLX, ws=GLW):
    m, r = 0.5 * (a + b), 0.5 * (b - a)
    return r * sum(w * fun(m + r * x) for x, w in zip(xs, ws))


# ------------------------------------------------------------------------------------------------------------
# 1. sieve functions on the grid s = i/N
# ------------------------------------------------------------------------------------------------------------
SMAX = 40


def lookup(arr, N, closed, closed_hi, limit):
    M = len(arr) - 1

    def fun(s):
        if s <= closed_hi:
            return closed(s)
        x = s * N
        i = int(x)
        if i >= M:
            return limit
        t = x - i
        return arr[i] + t * (arr[i + 1] - arr[i])
    return fun


def half_dim(N):
    """kappa = 1/2, beta = 1:  (sqrt(s) F)' = f(s-1)/(2 sqrt s) (s > 2),  (sqrt(s) f)' = F(s-1)/(2 sqrt s) (s > 3),
    F = AH/sqrt(s) on (0,2], f = (AH/sqrt s) log(sqrt s + sqrt(s-1)) on [1,3]  ([FI, (14.2)-(14.3)])."""
    M = SMAX * N
    F = [0.0] * (M + 1)
    f = [0.0] * (M + 1)
    Fc = lambda s: AH / math.sqrt(s)
    fc = lambda s: AH / math.sqrt(s) * math.log(math.sqrt(s) + math.sqrt(s - 1)) if s > 1 else 0.0
    for i in range(1, M + 1):
        s = i / N
        if s <= 2 + 1e-12:
            F[i] = Fc(s)
        if s <= 3 + 1e-12:
            f[i] = fc(s)
    G = math.sqrt(2.0) * F[2 * N]
    Hf = math.sqrt(3.0) * f[3 * N]
    for i in range(2 * N + 1, M + 1):
        s0, s1 = (i - 1) / N, i / N
        if s1 <= 4 + 1e-12:
            # t = 2 + u^2 makes f(t-1) = AH asinh(u)/sqrt(1+u^2) smooth:  dt = 2u du
            u0, u1 = math.sqrt(s0 - 2), math.sqrt(s1 - 2)
            G += gl(lambda u: (2 + u * u) ** -0.5 * AH * math.asinh(u) / math.sqrt(1 + u * u) * u,
                    u0, u1, GL5X, GL5W)
        else:
            G += 0.5 / N * (0.5 * f[i - 1 - N] / math.sqrt(s0) + 0.5 * f[i - N] / math.sqrt(s1))
        F[i] = G / math.sqrt(s1)
        if i > 3 * N:
            Hf += 0.5 / N * (0.5 * F[i - 1 - N] / math.sqrt(s0) + 0.5 * F[i - N] / math.sqrt(s1))
            f[i] = Hf / math.sqrt(s1)
    return (lookup(F, N, Fc, 2.0, 1.0), lookup(f, N, fc, 3.0, 1.0))


def linear_correct(N):
    """kappa = 1, beta = 2: (sF)' = f(s-1) (s > 3), (sf)' = F(s-1) (s > 4), F = 2e^g/s on (0,3],
    f = 2e^g log(s-1)/s on [2,4], f = 0 on (0,2]."""
    M = SMAX * N
    Fc = lambda s: 2 * EG / s
    fc = lambda s: 2 * EG * math.log(s - 1) / s if s > 2 else 0.0
    F = [0.0] + [Fc(i / N) if i <= 3 * N else 0.0 for i in range(1, M + 1)]
    f = [0.0] + [fc(i / N) if i <= 4 * N else 0.0 for i in range(1, M + 1)]
    sF = 3 * F[3 * N]
    sf = 4 * f[4 * N]
    for i in range(3 * N + 1, M + 1):
        sF += 0.5 / N * (f[i - 1 - N] + f[i - N])
        F[i] = sF / (i / N)
        if i > 4 * N:
            sf += 0.5 / N * (F[i - 1 - N] + F[i - N])
            f[i] = sf / (i / N)
    return (lookup(F, N, Fc, 3.0, 1.0), lookup(f, N, fc, 4.0, 1.0))


def linear_matlab(N):
    """as coded in Computation_files/{lower_f_sigma,upper_F_sigma}.m (solve_DDE_2): history F = 2e^g/s, f = 0 for
    s <= 2, then F' = f(s-1) - F, f' = F(s-1) - f for s > 2 (no factor 1/s)."""
    M = SMAX * N
    Fc = lambda s: 2 * EG / s
    fc = lambda s: 0.0
    F = [0.0] + [Fc(i / N) if i <= 2 * N else 0.0 for i in range(1, M + 1)]
    f = [0.0] * (M + 1)
    e = math.exp(-1.0 / N)
    for i in range(2 * N + 1, M + 1):
        F[i] = F[i - 1] * e + 0.5 / N * (e * f[i - 1 - N] + f[i - N])
        f[i] = f[i - 1] * e + 0.5 / N * (e * F[i - 1 - N] + F[i - N])
    return (lookup(F, N, Fc, 2.0, F[M]), lookup(f, N, fc, 2.0, f[M]))


def linear_printed(N):
    """as printed on p. 23: F = 2e^g/s on [1,3], f = 2e^g log(s-1)/s on [2,4], then F' = f(s-1) - F (s > 3),
    f' = F(s-1) - f (we start f at s = 4, where the printed initial segment ends)."""
    M = SMAX * N
    Fc = lambda s: 2 * EG / s
    fc = lambda s: 2 * EG * math.log(s - 1) / s if s > 2 else 0.0
    F = [0.0] + [Fc(i / N) if i <= 3 * N else 0.0 for i in range(1, M + 1)]
    f = [0.0] + [fc(i / N) if i <= 4 * N else 0.0 for i in range(1, M + 1)]
    e = math.exp(-1.0 / N)
    for i in range(3 * N + 1, M + 1):
        F[i] = F[i - 1] * e + 0.5 / N * (e * f[i - 1 - N] + f[i - N])
        if i > 4 * N:
            f[i] = f[i - 1] * e + 0.5 / N * (e * F[i - 1 - N] + F[i - N])
    return (lookup(F, N, Fc, 3.0, F[M]), lookup(f, N, fc, 4.0, f[M]))


NGRID = 1000 if QUICK else 2000
F1, f1 = half_dim(NGRID)
F2c, f2c = linear_correct(NGRID)
F2m, f2m = linear_matlab(NGRID)
F2p, f2p = linear_printed(NGRID)
F1h, f1h = half_dim(NGRID // 2)
F2h, f2h = linear_correct(NGRID // 2)

say("Audit of Nath-Xie, arXiv:2501.16723v3, Section 5 -- pure-Python recomputation (grid 1/%d)  %s" % (NGRID, clock()))
say("=" * 112)
say("\n1. SIEVE FUNCTIONS")
say("   half-dimensional (kappa=1/2, beta=1): printed initial segments agree with [FI, (14.2)-(14.3)];")
say("   the printed f_1 = sqrt(e^g/(pi s)) log(1+2(s-1)+2 sqrt(s(s-1))) equals (e^g/(pi s))^{1/2} int_1^s dt/sqrt(t(t-1)):")
mx = max(abs(math.sqrt(EG / (math.pi * s)) * math.log(1 + 2 * (s - 1) + 2 * math.sqrt(s * (s - 1)))
             - math.sqrt(EG / (math.pi * s)) * gl(lambda u: 2.0 / math.sqrt(1 + u * u), 0, math.sqrt(s - 1)))
         for s in [1 + k / 50 for k in range(1, 101)])
say("   max |printed - integral form| on [1,3] = %.2e  (so no factor-2 error in f_1)" % mx)
say("   ratio f_1/F_1 = log(sqrt s + sqrt(s-1)) on [1,2]; at s = 1/(2*0.4230) = %.6f it is %.6f"
    % (1 / 0.846, f1(1 / 0.846) / F1(1 / 0.846)))
say("   closed-form checks of the linear functions: F2(4) = (e^g/2)(1 + int_2^3 log(t-1)/t dt) = %.7f, grid %.7f;"
    % (EG / 2 * (1 + gl(lambda t: math.log(t - 1) / t, 2, 3)), F2c(4)))
say("     F2(5) = (2e^g/5)(1 + int_2^4 log(t-1)/t dt) = %.7f, grid %.7f;  f2(4) = e^g log(3)/2 = %.7f"
    % (2 * EG / 5 * (1 + gl(lambda t: math.log(t - 1) / t, 2, 4)), F2c(5), EG * math.log(3) / 2))
errh = max(max(abs(F1(s) - F1h(s)), abs(f1(s) - f1h(s)), abs(F2c(s) - F2h(s)), abs(f2c(s) - f2h(s)))
           for s in [2 + k / 20 for k in range(0, 161)])
say("   grid check: max difference between steps 1/%d and 1/%d on [2,10] = %.1e" % (NGRID, NGRID // 2, errh))
say("")
say("     s   |  F1(s)     f1(s)   | F2 correct  f2 correct | F2 printed  f2 printed | F2 Matlab   f2 Matlab")
for s in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0):
    say("   %5.2f | %8.5f  %8.5f | %8.5f  %8.5f   | %8.5f  %8.5f   | %8.5f  %8.5f"
        % (s, F1(s), f1(s), F2c(s), f2c(s), F2p(s), f2p(s), F2m(s), f2m(s)))
say("   (Matlab system on [2,3]: F2 = e^g e^{2-s}, f2 = e^{-s} int_2^s 2e^g e^t/(t-1) dt; e.g. F2(3) = %.5f, f2(3) = %.5f:"
    % (EG * math.exp(-1), math.exp(-3) * gl(lambda t: 2 * EG * math.exp(t) / (t - 1), 2, 3)))
say("    the 'lower' function exceeds the 'upper' one, f2 > F2 on (%.3f, %.3f) approximately)" % (
    min(s / 100 for s in range(201, 1500) if f2m(s / 100) > F2m(s / 100)),
    max(s / 100 for s in range(201, 1500) if f2m(s / 100) > F2m(s / 100))))
say("   %s" % clock())


# ------------------------------------------------------------------------------------------------------------
# 2. constants
# ------------------------------------------------------------------------------------------------------------
def prime_list(n):
    sv = bytearray([1]) * (n + 1)
    sv[0] = sv[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if sv[i]:
            sv[i * i::i] = bytearray(len(range(i * i, n + 1, i)))
    return [i for i, b in enumerate(sv) if b]


PMAX = 10 ** 6 if QUICK else 10 ** 7
PR = prime_list(PMAX)
P1 = [p for p in PR if p % 4 == 1]
P3 = [p for p in PR if p % 4 == 3]


def prod_log(it):
    return math.exp(math.fsum(math.log(v) for v in it))


C_ = 2.25 * prod_log(1 - (3 * p - 1) / (p - 1) ** 3 for p in P3 if p > 3) * prod_log(1 - 1 / (p - 1) ** 2 for p in P1)
C_typo = 2.25 * prod_log(1 - (3 * p - 1) / (p - 1) ** 3 for p in P3 if p > 3) * prod_log(1 - 1 / (p - 2) ** 2 for p in P1)
K3 = prod_log(1 - 1 / p ** 2 for p in P3)
KAPPA = math.sqrt(math.pi / 2 * K3)
c1 = prod_log(1 + 1 / (p - 2) ** 2 for p in PR if p > 3)
c1_joint = prod_log(1 - 1 / (p - 2) ** 2 for p in PR if p > 3)
c2 = prod_log(1 - 1 / (p - 1) ** 2 for p in PR if p > 2)
P1prod = prod_log((1 - 1 / p) * (1 + p * p / (p - 1) ** 3) for p in P1)
c3_print = math.sqrt(K3) * P1prod                      # value of the printed (conditionally convergent) product
c3_true = c3_print / math.sqrt(2)                      # Wirsing constant: includes (1 - 1/2)^{1/2}


def kp(p):
    return (1 - 1 / (p - 1) ** 2) ** 2 * (1 - 1 / (p - 2) ** 2)


c3_joint = math.sqrt(K3 / 2) * prod_log((1 - 1 / p) * (1 + p * p / ((p - 1) ** 3 * kp(p))) for p in P1)
# exactly as in the Matlab files
PR4 = [p for p in PR if p <= 10 ** 4]
PR6 = [p for p in PR if p <= 10 ** 6]
C_m = 2.25 * prod_log(1 - (3 * p - 1) / (p - 1) ** 3 for p in PR6 if p % 4 == 3 and p > 3) * prod_log(
    1 - 1 / (p - 1) ** 2 for p in PR6 if p % 4 == 1)
KAPPA_m = math.sqrt(math.pi / 2 * prod_log(1 - 1 / p ** 2 for p in PR6 if p % 4 == 3))
c1_m = prod_log(1 + 1 / (p - 2) ** 2 for p in PR4 if p > 3)
c2_m = prod_log(1 - 1 / (p - 1) ** 2 for p in PR4 if p > 2)
c3_m = 2 / math.sqrt(math.pi) * prod_log((1 - 1 / p) ** 0.5 * (1 + p * p / (p - 1) ** 3) for p in PR4 if p % 4 == 1) \
    * prod_log((1 - 1 / p) ** 0.5 for p in PR4 if p % 4 == 3)
AMAIN = C_ * KAPPA * math.exp(-1.5 * GAMMA) / 4
AMAIN_m = C_m * KAPPA_m * math.exp(-1.5 * GAMMA) / 4
SW_print = 2 * c1 * c2 ** 2 * c3_print
SW_m = 2 * c1_m * c2_m ** 2 * c3_m
SW_corr = 2 * c1_joint * c2 ** 2 * c3_joint

say("\n2. CONSTANTS (primes up to %.0e; tails < 1e-6)  %s" % (PMAX, clock()))
say("   C     = %.6f   (Prop. 4.1, (4.6), Sect. 5: factor (1 - 1/(p-1)^2) for p = 1 (4))" % C_)
say("   C'    = %.6f   (Prop. 4.3 as printed, (p-2)^2: a misprint; (4.5) gives (p-1)^2)" % C_typo)
say("   kappa = %.6f,  c1 = %.6f,  c2 = %.6f" % (KAPPA, c1, c2))
say("   c3 as printed: the products over p = 1 (4) and p = 3 (4) diverge separately (factors 1 +- 1/(2p) + O(p^-2));")
say("      in the natural order the printed product equals prod_{p=3(4)}(1-p^-2)^{1/2} prod_{p=1(4)}(1-1/p)(1+p^2/(p-1)^3) = %.6f" % c3_print)
say("   c3 (Wirsing / [Martin, Prop. A.3], with the factor (1 - 1/2)^{1/2} of p = 2) = c3_print/sqrt 2 = %.6f" % c3_true)
say("   joint singular series of Prop. 4.2: per prime p > 3, p not dividing m: (1-1/(p-1)^2)^2 (1-1/(p-2)^2); p | m: 1")
say("      printed bound uses (1+1/(p-2)^2)(1-1/(p-1)^2)^2 for all p > 3 (c1 c2^2); H_1(m) <= (2m/phi(2m)) c2 is reversed")
say("      corrected: prod_{p>3}(1-(p-2)^-2) = %.6f, c3'' (m-weights 1/k_p, k_p = (1-(p-1)^-2)^2(1-(p-2)^-2)) = %.6f" % (c1_joint, c3_joint))
say("   switched-term coefficient 2 c1 c2^2 c3:  printed %.6f   Matlab %.6f   corrected %.6f   (ratio corrected/printed %.4f)"
    % (SW_print, SW_m, SW_corr, SW_corr / SW_print))
say("   main prefactor A = C kappa e^{-3g/2}/4 = %.6f  (Matlab constants: %.6f)" % (AMAIN, AMAIN_m))
say("   Matlab constants: C %.6f kappa %.6f c1 %.6f c2 %.6f c3 %.6f (primes to 1e6, 1e6, 1e4, 1e4, 1e4)"
    % (C_m, KAPPA_m, c1_m, c2_m, c3_m))


# ------------------------------------------------------------------------------------------------------------
# 3. direct check of c3
# ------------------------------------------------------------------------------------------------------------
def wirsing_check(T):
    """S(t) = sum_{m <= t, p | m => p = 1 (4)} m/phi(m)^2, and the same with weights prod_{p | m} 1/k_p."""
    ps = [p for p in P1 if p <= T]
    pts = []

    def dfs(start, n, w, w2):
        pts.append((n, w, w2))
        for j in range(start, len(ps)):
            p = ps[j]
            if n * p > T:
                break
            q = p
            phi_ratio = (p / (p - 1)) ** 2       # (n/phi(n))^2 factor for the prime p
            while n * q <= T:
                nn = n * q
                dfs(j + 1, nn, w * phi_ratio, w2 * phi_ratio / kp(p))
                q *= p
    dfs(0, 1, 1.0, 1.0)
    pts.sort()
    res = []
    acc = acc2 = 0.0
    k = 0
    marks = [10 ** e for e in range(3, int(round(math.log10(T))) + 1)]
    for mk in marks:
        while k < len(pts) and pts[k][0] <= mk:
            n, w, w2 = pts[k]
            acc += w / n
            acc2 += w2 / n
            k += 1
        res.append((mk, acc, acc2))
    return res


TW = 10 ** 6 if QUICK else 10 ** 7
say("\n3. DIRECT CHECK OF c3: S2(t) = sum b*(m) m/phi(m)^2 ~ c3 (log t)^{1/2}  (Martin, Prop. A.3)  %s" % clock())
res = wirsing_check(TW)
say("        t      S2(t)    S2/sqrt(log t)   |  S2 with 1/k_p weights   ratio")
for t, a, a2 in res:
    L = math.log(t)
    say("   %8.0e  %9.5f   %9.5f        |  %9.5f              %9.5f" % (t, a, a / math.sqrt(L), a2, a2 / math.sqrt(L)))
(tA, a1, b1), (tB, a2_, b2_) = res[-2], res[-1]
LA, LB = math.log(tA), math.log(tB)
det = math.sqrt(LA) / math.sqrt(LB) - math.sqrt(LB) / math.sqrt(LA)
fitc = (a1 / math.sqrt(LB) - a2_ / math.sqrt(LA)) / det
fitc2 = (b1 / math.sqrt(LB) - b2_ / math.sqrt(LA)) / det
say("   two-term fit S2 = c sqrt(L) + d/sqrt(L) through the last two points: c = %.4f (prediction c3 = %.4f; printed %.4f)"
    % (fitc, c3_true, c3_print))
say("   same with the 1/k_p weights: c = %.4f (prediction c3'' = %.4f)  %s" % (fitc2, c3_joint, clock()))


# ------------------------------------------------------------------------------------------------------------
# 4. vector-sieve functions and the functionals of Section 5
# ------------------------------------------------------------------------------------------------------------
INVPHI = (math.sqrt(5) - 1) / 2


def golden(fun, a, b, it=40, maximize=True):
    sg = 1 if maximize else -1
    c, d = b - INVPHI * (b - a), a + INVPHI * (b - a)
    fc, fd = sg * fun(c), sg * fun(d)
    for _ in range(it):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - INVPHI * (b - a)
            fc = sg * fun(c)
        else:
            a, c, fc = c, d, fd
            d = a + INVPHI * (b - a)
            fd = sg * fun(d)
    x = 0.5 * (a + b)
    return x, fun(x)


def fvec(X, t1, t2, F2, f2, n=240, profile=False):
    """f(sigma1, sigma2) of (3.14)/(5.1): sup over theta1 s1 + theta2 s2 = X, s1 >= 1, s2 >= 2 of
    f1(s1) F2(s2) + f2(s2) F1(s1) - F1(s1) F2(s2).  Returns (value, s1, s2) or None if the set is empty."""
    hi = (X - 2 * t2) / t1
    if hi < 1:
        return None

    def obj(s1):
        s2 = (X - t1 * s1) / t2
        a, b = F1(s1), F2(s2)
        return f1(s1) * b + f2(s2) * a - a * b
    if hi == 1:
        return obj(1.0), 1.0, 2.0
    g = [1 + (hi - 1) * k / n for k in range(n + 1)]
    v = [obj(s) for s in g]
    if profile:
        return g, v
    k = max(range(n + 1), key=lambda j: v[j])
    a, b = g[max(k - 1, 0)], g[min(k + 1, n)]
    x, val = golden(obj, a, b, 30, True)
    if v[k] > val:
        x, val = g[k], v[k]
    return val, x, (X - t1 * x) / t2


def Fvec(X, t1, t2, F2, n=160, box="paper"):
    """F(sigma1, sigma2) of (3.13)/(5.1): inf over theta1 s1 + theta2 s2 = X of F1(s1) F2(s2), with s1 > 0, s2 >= 1
    (paper) or s1 >= 1, s2 >= 2 (Matlab).  None if the set is empty."""
    if box == "paper":
        lo, hi = 1e-9, (X - t2) / t1
    else:
        lo, hi = 1.0, (X - 2 * t2) / t1
    if hi <= lo:
        return None

    def obj(s1):
        return F1(s1) * F2((X - t1 * s1) / t2)
    # most of the action is at small s1 when theta2 is small: geometric grid
    if box == "paper":
        g = [hi * (1e-6) ** (1 - k / n) for k in range(n + 1)]
    else:
        g = [lo + (hi - lo) * k / n for k in range(n + 1)]
    v = [obj(s) for s in g]
    k = min(range(n + 1), key=lambda j: v[j])
    a, b = g[max(k - 1, 0)], g[min(k + 1, n)]
    x, val = golden(obj, a, b, 30, False)
    return min(val, v[k])


def C_theta(t1, matlab=False):
    """C(theta1) = int_0^{1-2 theta1} log((1-b-theta1)/theta1) db / (b^{1/2}(1-b)), b = u^2.  Matlab: (1-b)^{1/2}."""
    e = 0.5 if matlab else 1.0
    return gl(lambda u: 2 * math.log((1 - u * u - t1) / t1) / (1 - u * u) ** e, 0, math.sqrt(1 - 2 * t1))


def alpha_table(t1, t2, F2, amax, n=60, box="paper"):
    """nodes alpha_j (log-spaced on [theta2, amax]) and the values F((1-2a)/(2 t1), (1-2a)/(2 t2))"""
    la, lb = math.log(t2), math.log(amax)
    al = [math.exp(la + (lb - la) * k / n) for k in range(n + 1)]
    return al, [Fvec(0.5 - a, t1, t2, F2, box=box) for a in al]


def I_from_table(al, Fv, th):
    """I(theta) = int_{theta2}^{theta} (1 - a/theta) F(...) da/a  (trapezoid in log a; theta on or between nodes)"""
    tot = 0.0
    for j in range(len(al) - 1):
        a0, a1 = al[j], al[j + 1]
        if a0 >= th:
            break
        if Fv[j] is None or Fv[j + 1] is None:
            continue
        if a1 > th:      # partial last cell, linear in log a
            tot += 0.5 * (math.log(th) - math.log(a0)) * (1 - a0 / th) * Fv[j]     # integrand vanishes at a = theta
            break
        tot += 0.5 * (math.log(a1) - math.log(a0)) * ((1 - a0 / th) * Fv[j] + (1 - a1 / th) * Fv[j + 1])
    return tot


def I_direct(th, t1, t2, F2, box="paper"):
    """I by Gauss-Legendre in log alpha (40 nodes on each of 4 panels)"""
    la, lb = math.log(t2), math.log(th)
    tot = 0.0
    for k in range(4):
        a, b = la + (lb - la) * k / 4, la + (lb - la) * (k + 1) / 4

        def g(u):
            al = math.exp(u)
            v = Fvec(0.5 - al, t1, t2, F2, box=box)
            return 0.0 if v is None else (1 - al / th) * v
        tot += gl(g, a, b)
    return tot


WCACHE = {}


def Wlev(t1, t2, mode):
    """1/(log D1 log D2) in units of (log x)^-2, times the factor e^{-g} F2(s2)/2 of the second upper sieve"""
    if mode == "printed":
        return 1.0 / (t2 * min(t1, 0.5 - t2))
    if t2 in WCACHE:
        return WCACHE[t2]
    best = None
    for k in range(0, 4001):
        s2 = 1 + k * 0.005
        if s2 * t2 >= 0.49:
            break
        w = (math.exp(-GAMMA) * F2c(s2) / 2) / (t2 * (0.5 - s2 * t2))
        if best is None or w < best[0]:
            best = (w, s2)
    WCACHE[t2] = best[0]
    return best[0]


def Wlev_arg(t2):
    best = None
    for k in range(0, 4001):
        s2 = 1 + k * 0.005
        if s2 * t2 >= 0.49:
            break
        w = (math.exp(-GAMMA) * F2c(s2) / 2) / (t2 * (0.5 - s2 * t2))
        if best is None or w < best[0]:
            best = (w, s2)
    return best[1]


say("\n4. THE FUNCTIONALS OF SECTION 5 AT THE AUTHORS' POINTS  %s" % clock())
t1, t2 = 0.4230, 0.0249
s1g, s2g = 1 / (2 * t1), 1 / (2 * t2)
say("   (a) G(0.4230, 0.0249): sigma1 = %.6f, sigma2 = %.6f; the constraint s1/sigma1 + s2/sigma2 = 1 with s1 >= 1, s2 >= 2"
    % (s1g, s2g))
say("       forces s1 in [1, %.4f], s2 in [2, %.4f]" % ((0.5 - 2 * t2) / t1, (0.5 - t1) / t2))
say("       brief's 'direct' value f1(s1g)F2(s2g)+F1(s1g)f2(s2g)-F1F2 (not admissible: s1/s1g + s2/s2g = 2) = %.5f"
    % (f1(s1g) * F2c(s2g) + F1(s1g) * f2c(s2g) - F1(s1g) * F2c(s2g)))
gv, vv = fvec(0.5, t1, t2, F2c, f2c, n=400, profile=True)
say("       over the admissible segment, correct functions: inf = %.5f, sup = %.5f;   Matlab functions: inf = %.5f, sup = %.5f"
    % (min(vv), max(vv), min(fvec(0.5, t1, t2, F2m, f2m, n=400, profile=True)[1]),
       max(fvec(0.5, t1, t2, F2m, f2m, n=400, profile=True)[1])))
rows = []
for lab, F2, f2, A, SW, Cf in (("Matlab functions + Matlab constants", F2m, f2m, AMAIN_m, SW_m, lambda t: C_theta(t, True)),
                               ("printed system (segments, no 1/s)", F2p, f2p, AMAIN, SW_print, C_theta),
                               ("correct functions, printed constants", F2c, f2c, AMAIN, SW_print, C_theta),
                               ("correct functions, corrected consts", F2c, f2c, AMAIN, SW_corr, C_theta)):
    fv = fvec(0.5, t1, t2, F2, f2)
    main = A * fv[0] / (math.sqrt(t1) * t2)
    sw = SW * Cf(t1) / (t2 * min(t1, 0.5 - t2))
    say("       %-38s f = %8.5f at (s1,s2) = (%.4f, %.4f): main %8.4f  switched %7.4f  G = %9.5f"
        % (lab, fv[0], fv[1], fv[2], main, sw, main - sw))
say("       (paper: G(0.4230, 0.0249) = 0.001378)")
say("       C(theta1) at 0.4230: paper's integrand %.6f, Matlab integrand 1/sqrt(b(1-b)) %.6f" % (C_theta(t1), C_theta(t1, True)))


def maxG(t2, F2, f2, A, SW, Cf, step=0.001, start=0.251, first=False):
    best = None
    t1 = start
    while t1 + 2 * t2 <= 0.5 + 1e-12:
        fv = fvec(0.5, t1, t2, F2, f2, n=80)
        if fv is not None:
            g = A * fv[0] / (math.sqrt(t1) * t2) - SW * Cf(t1) / (t2 * min(t1, 0.5 - t2))
            if first and g > 0:
                return g, t1
            if best is None or g > best[0]:
                best = (g, t1)
        t1 = round(t1 + step, 6)
    return best


# replicate the search of find_theta_values_G.m (theta2 from 0.04 down by 1e-4; theta1 from 0.251 up by 1e-3)
Cm = lambda t: C_theta(t, True)
lo, hi = 0.0001, 0.04
for _ in range(9):
    mid = 0.5 * (lo + hi)
    g = maxG(mid, F2m, f2m, AMAIN_m, SW_m, Cm, step=0.004)
    if g is not None and g[0] > 0:
        lo = mid
    else:
        hi = mid
t2s = round(lo, 4) + 0.0003
while t2s > 0.0001:
    g = maxG(t2s, F2m, f2m, AMAIN_m, SW_m, Cm, step=0.001, first=True)
    if g is not None and g[0] > 0:
        break
    t2s = round(t2s - 0.0001, 4)
say("   (b) replication of find_theta_values_G.m with the Matlab functions and constants (sup on a grid instead of fmincon):")
say("       largest theta2 (step 1e-4) with some theta1 (step 1e-3 from 0.251) giving G > 0: theta2 = %.4f, first theta1 = %.3f,"
    " G = %.6f" % (t2s, g[1], g[0]))
say("       (paper: theta2 = 0.0249, theta1 = 0.4230, G = 0.001378)  %s" % clock())

# H at (0.17, 0.25, 0.449, 0.011)
lam, th, t1, t2 = 0.17, 0.25, 0.449, 0.011
say("   (c) H(0.17, 0.25, 0.449, 0.011); admissible s1 in [1, %.4f], s2 in [2, %.4f]" % ((0.5 - 2 * t2) / t1, (0.5 - t1) / t2))
say("       Matlab box s1 >= 1, s2 >= 2 for F(sigma1(a), sigma2(a)) is empty for a > %.4f, so compute_I.m evaluates fmincon"
    " at infeasible points there;" % (0.5 - t1 - 2 * t2))
say("       below, I uses the paper's (5.1) (s1 > 0, s2 >= 1)")
HROWS = {}
for lab, F2, f2, A, SW, Cf in (("Matlab functions + Matlab constants", F2m, f2m, AMAIN_m, SW_m, lambda t: C_theta(t, True)),
                               ("correct functions, printed constants", F2c, f2c, AMAIN, SW_print, C_theta),
                               ("correct functions, corrected consts", F2c, f2c, AMAIN, SW_corr, C_theta)):
    fv = fvec(0.5, t1, t2, F2, f2)
    I = I_direct(th, t1, t2, F2)
    main = A * fv[0] / (math.sqrt(t1) * t2)
    sw = SW * Cf(t1) / (t2 * min(t1, 0.5 - t2))
    rich = lam * A * I / (math.sqrt(t1) * t2)
    HROWS[lab] = (fv[0], I, main, sw, rich, main - sw - rich)
    say("       %-38s f = %8.5f  I = %7.4f | main %8.4f  switched %7.4f  lambda*Richert %7.4f  H = %9.4f"
        % (lab, fv[0], I, main, sw, rich, main - sw - rich))
say("       (paper: H = 1.2690)")
g_, v_ = fvec(0.5, t1, t2, F2m, f2m, n=2000, profile=True)
locs = [(g_[k], v_[k]) for k in range(1, len(v_) - 1) if v_[k] >= v_[k - 1] and v_[k] >= v_[k + 1]]
say("       objective along the segment with the Matlab functions: %d local maxima %s; value at s1 = 1: %.4f, at s1 = max: %.4f"
    % (len(locs), ", ".join("(s1 %.4f: %.4f)" % l for l in locs[:6]), v_[0], v_[-1]))
fM, IM_paper = HROWS["Matlab functions + Matlab constants"][0], HROWS["Matlab functions + Matlab constants"][1]
mainM, swM = HROWS["Matlab functions + Matlab constants"][2], HROWS["Matlab functions + Matlab constants"][3]
Ineed = (mainM - swM - 1.2690) * math.sqrt(t1) * t2 / (lam * AMAIN_m)
say("       I that reproduces H = 1.2690 with f = %.5f: %.4f" % (fM, Ineed))
# hypothesis: where the box s1 >= 1, s2 >= 2 is empty, fmincon stops at the bound point (1, 2) (least violation of the
# equality constraint, all of whose coefficients are positive) and upper_F_sigma returns F1(1) F2(2)
F12 = F1(1.0) * F2m(2.0)


def I_matlab(th, t1, t2):
    la, lb = math.log(t2), math.log(th)
    tot = 0.0
    cut = 0.5 - t1 - 2 * t2
    for (a, b) in ((la, math.log(cut)), (math.log(cut), lb)):
        def g(u):
            al = math.exp(u)
            v = Fvec(0.5 - al, t1, t2, F2m, box="matlab")
            return (1 - al / th) * (F12 if v is None else v)
        for k in range(4):
            tot += gl(g, a + (b - a) * k / 4, a + (b - a) * (k + 1) / 4)
    return tot


IM = I_matlab(th, t1, t2)
HM = mainM - swM - lam * AMAIN_m * IM / (math.sqrt(t1) * t2)
say("       Matlab box with F = F1(1) F2(2) = %.5f where the box is empty: I = %.4f, H = %.4f  (paper 1.2690)" % (F12, IM, HM))
HMATLAB = (IM, HM)
say("   %s" % clock())


# ------------------------------------------------------------------------------------------------------------
# 5. what the method gives with the correct sieve functions
# ------------------------------------------------------------------------------------------------------------
say("\n5. THE METHOD WITH THE CORRECT SIEVE FUNCTIONS")
say("   R0: constants as printed (c3 = %.4f, c1 = %.4f);  R1: corrected constants (coefficient x %.4f);"
    % (c3_print, c1, SW_corr / SW_print))
say("   R2: R1 and the levels of Prop. 4.2 chosen freely: D2 = z2^{s2}, D1 = x^{1/2}/D2 (only 4 D1 D2 <= x^{1/2} (log x)^-B is used)")
say("       i.e. 1/(theta2 min(theta1, 1/2-theta2)) -> min_{s2>=1} (e^-g F2(s2)/2)/(theta2 (1/2 - s2 theta2))")
VARIANTS = (("R0", SW_print, "printed"), ("R1", SW_corr, "printed"), ("R2", SW_corr, "free"))
FCACHE = {}


def fval(t1, t2):
    key = (round(t1, 6), round(t2, 7))
    if key not in FCACHE:
        fv = fvec(0.5, t1, t2, F2c, f2c, n=120)
        FCACHE[key] = None if fv is None else fv[0]
    return FCACHE[key]


def Gfun(t1, t2, SW, mode):
    fv = fval(t1, t2)
    if fv is None:
        return None
    return AMAIN * fv / (math.sqrt(t1) * t2) - SW * C_theta(t1) * Wlev(t1, t2, mode)


def maxG_c(t2, SW, mode):
    best = None
    t1 = 0.30
    while t1 + 2 * t2 <= 0.5 + 1e-12:
        g = Gfun(t1, t2, SW, mode)
        if g is not None and (best is None or g > best[0]):
            best = (g, t1)
        t1 = round(t1 + 0.0025, 6)
    if best is None:
        return None
    # refine theta1
    a, b = max(0.30, best[1] - 0.0025), min(best[1] + 0.0025, 0.5 - 2 * t2)
    def gsafe(u):
        v = Gfun(u, t2, SW, mode)
        return -1e9 if v is None else v
    x, v = golden(gsafe, a, b, 30, True)
    return max(best, (v, x))


say("\n   (a) unweighted (G > 0): largest theta2 for which some theta1 gives G > 0, and omega(p+2) <= floor(1/theta2)")
UNW = {}
for name, SW, mode in VARIANTS:
    lo, hi = 1e-4, 0.03
    if (maxG_c(lo, SW, mode) or (-1,))[0] <= 0:
        say("       %s: no theta2 >= 1e-4 works" % name)
        continue
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        g = maxG_c(mid, SW, mode)
        if g is not None and g[0] > 0:
            lo = mid
        else:
            hi = mid
    g = maxG_c(lo, SW, mode)
    UNW[name] = (lo, g[1])
    say("       %s: theta2* = %.6f (theta1 = %.4f), 1/theta2* = %.2f  =>  omega(p+2) <= %d   %s"
        % (name, lo, g[1], 1 / lo, math.floor(1 / lo), clock()))

say("\n   (b) weighted (H > 0): minimum of 1/lambda + 1/theta; omega(p+2) < 1/lambda + 1/theta")
TH2 = [0.0004, 0.0005, 0.0006, 0.0008, 0.001, 0.0012, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010, 0.012, 0.015]
if QUICK:
    TH2 = [0.001, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.010]
ATAB = {}


def best_weighted(t1, t2, SW, mode, refine=False):
    fv = fval(t1, t2)
    if fv is None:
        return None
    main = AMAIN * fv / (math.sqrt(t1) * t2)
    sw = SW * C_theta(t1) * Wlev(t1, t2, mode)
    if main <= sw:
        return None
    amax = min(0.45, 0.5 - t2 - 0.005)
    key = (round(t1, 6), round(t2, 7), refine)
    if key not in ATAB:
        ATAB[key] = alpha_table(t1, t2, F2c, amax, n=(160 if refine else 40))
    al, Fv = ATAB[key]
    best = None
    for th in al[2:]:
        I = I_from_table(al, Fv, th)
        rich = AMAIN * I / (math.sqrt(t1) * t2)
        if rich <= 0:
            continue
        lam = (main - sw) / rich
        V = 1 / lam + 1 / th
        if best is None or V < best[0]:
            best = (V, th, lam, main, sw, rich * lam)
    return best


RES = {}
for name, SW, mode in VARIANTS:
    glob = None
    for t2 in TH2:
        t1 = 0.30
        while t1 + 2 * t2 <= 0.5 + 1e-12:
            b = best_weighted(t1, t2, SW, mode)
            if b and (glob is None or b[0] < glob[0]):
                glob = b + (t1, t2)
            t1 = round(t1 + 0.0025, 6)
    if glob is None:
        say("       %s: H > 0 nowhere in the scan" % name)
        continue
    V, th, lam, main, sw, rl, t1, t2 = glob
    bb = best_weighted(t1, t2, SW, mode, refine=True)
    V, th, lam, main, sw, rl = bb
    I_chk = I_direct(th, t1, t2, F2c)
    RES[name] = (V, th, lam, t1, t2, main, sw, rl)
    say("       %s: min 1/lam + 1/theta = %.3f  =>  omega(p+2) <= %d   at theta1 %.4f theta2 %.4f theta %.4f lambda %.5f"
        % (name, V, math.ceil(V) - 1, t1, t2, th, lam))
    say("           main %.4f  switched %.4f  lambda*Richert %.4f;  I (table) %.5f, I (Gauss) %.5f  %s"
        % (main, sw, rl, rl / lam * math.sqrt(t1) * t2 / AMAIN, I_chk, clock()))
if "R2" in RES:
    say("       (R2: the optimal s2 in Prop. 4.2 at theta2 = %.4f is %.3f)" % (RES["R2"][4], Wlev_arg(RES["R2"][4])))

say("\n   summary: with the correct linear-sieve functions the printed point gives H < 0; within the paper's method")
for name in ("R0", "R1", "R2"):
    if name in RES:
        say("     %s: Omega(p+2) <= %d (weighted), %s" % (name, math.ceil(RES[name][0]) - 1,
            ("<= %d unweighted" % math.floor(1 / UNW[name][0])) if name in UNW else "unweighted: none"))
say("\ndone %s" % clock())

if __name__ == "__main__":
    with open(os.path.join(OUTDIR, "verify_nath_xie.txt"), "w", encoding="utf8") as fh:
        fh.write("\n".join(_out) + "\n")
