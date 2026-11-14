#!/usr/bin/env python3
"""Figures for nath_xie_sieve_audit.tex (matplotlib).  Usage: python code/make_figures.py
figures/fig_functions.pdf : (a) linear sieve functions, correct vs. as coded; (b) G(theta1, 0.0249)
figures/fig_restore.pdf   : min over theta1 of 1/lambda + 1/theta (H >= 0) as a function of theta2, variants R0-R2"""
import os, sys, io, contextlib, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = [sys.argv[0], "--quick"]
sys.path.insert(0, HERE)
with contextlib.redirect_stdout(io.StringIO()):
    import verify_nath_xie as V           # runs the (quick) recomputation silently
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"      # reference categorical slots 1-3
INK, MUTED = "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "lines.linewidth": 1.6})

# ---------------- figure 1 ----------------
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9))
S = [1 + k / 200 for k in range(0, 1401)]
ax.plot(S, [V.F2c(s) for s in S], color=C1, ls="-", label=r"$F_2$ correct")
ax.plot(S, [V.f2c(s) for s in S], color=C1, ls="--", label=r"$f_2$ correct")
ax.plot(S, [V.F2m(s) for s in S], color=C2, ls="-", label=r"$F_2$ Matlab")
ax.plot(S, [V.f2m(s) for s in S], color=C2, ls="--", label=r"$f_2$ Matlab")
ax.axhline(1, color=MUTED, lw=0.6)
ax.set_xlim(1, 8); ax.set_ylim(0, 3.6)
ax.set_xlabel(r"$s$"); ax.set_title("(a) linear sieve functions", fontsize=9, loc="left")
ax.legend(fontsize=7.5, loc="upper right")

T1 = [0.30 + k * 0.0025 for k in range(0, 61)]
t2 = 0.0249
for lab, F2, f2, A, SW, Cf, col, ls in (
        ("Matlab functions", V.F2m, V.f2m, V.AMAIN_m, V.SW_m, lambda t: V.C_theta(t, True), C2, "-"),
        ("correct functions", V.F2c, V.f2c, V.AMAIN, V.SW_print, V.C_theta, C1, "-")):
    xs, ys = [], []
    for t1 in T1:
        fv = V.fvec(0.5, t1, t2, F2, f2, n=150)
        if fv is None:
            continue
        xs.append(t1)
        ys.append(A * fv[0] / (math.sqrt(t1) * t2) - SW * Cf(t1) / (t2 * min(t1, 0.5 - t2)))
    bx.plot(xs, ys, color=col, ls=ls, label=lab)
bx.axhline(0, color=MUTED, lw=0.6)
bx.axvline(0.4230, color=MUTED, lw=0.6, ls=":")
bx.text(0.4245, 4, r"$\theta_1=0.4230$", fontsize=7.5, color=INK)
bx.set_xlabel(r"$\theta_1$"); bx.set_ylim(-40, 8)
bx.set_title(r"(b) $G(\theta_1, 0.0249)$", fontsize=9, loc="left")
bx.legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_functions.pdf"))

# ---------------- figure 2 ----------------
fig, ax = plt.subplots(figsize=(5.2, 3.0))
TH2 = [0.0004, 0.0005, 0.0006, 0.0008, 0.001, 0.0012, 0.0015, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.007,
       0.008, 0.010, 0.012, 0.015, 0.02]
for (name, SW, mode), col, ls in zip(V.VARIANTS, (C2, C3, C1), ("-", "--", "-")):
    xs, ys = [], []
    for t2 in TH2:
        best = None
        t1 = 0.30
        while t1 + 2 * t2 <= 0.5 + 1e-12:
            b = V.best_weighted(t1, t2, SW, mode)
            if b and (best is None or b[0] < best):
                best = b[0]
            t1 = round(t1 + 0.0025, 6)
        if best is not None:
            xs.append(t2)
            ys.append(best)
    ax.plot(xs, ys, color=col, ls=ls, marker="o", ms=3, label=name)
    k = min(range(len(ys)), key=lambda i: ys[i])
    ax.annotate("%s: %.1f" % (name, ys[k]), (xs[k], ys[k]), xytext=(4, -12), textcoords="offset points",
                fontsize=7.5, color=INK)
ax.axhline(9.88, color=MUTED, lw=0.6, ls=":")
ax.text(0.0105, 10.8, "claimed 9.88", fontsize=7.5, color=MUTED)
ax.set_xscale("log")
ax.set_ylim(0, 160)
ax.set_xlabel(r"$\theta_2$")
ax.set_ylabel(r"$\min_{\theta_1,\theta,\lambda}\ (1/\lambda+1/\theta)$")
ax.legend(fontsize=7.5, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_restore.pdf"))
fig.savefig(os.path.join(FIG, "fig_restore.png"), dpi=130)
plt.close("all")
fig = None
print("figures written to", os.path.normpath(FIG))
