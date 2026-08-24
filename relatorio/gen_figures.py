#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gera todas as figuras do relatório LaTeX."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fountain_sim_performance import simulate_ramsey_fringe, profile_envelope

BASE = "/home/felipe/Documents/ICs/Fountain/Simulações/fountain_sim"
FIG = BASE + "/relatorio/figuras"
NU0 = 9192631770.0
B = 3e-9

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

# ---------------------------------------------------------------------------
# Dados experimentais
# ---------------------------------------------------------------------------
df = pd.read_csv(BASE + "/data/exp_data.csv")
dv = df["Frequency"].values - NU0
y_raw = df["Prob."].values


def local_amp(x, w=8):
    o = np.zeros_like(x)
    for i in range(len(x)):
        lo = max(0, i - w)
        hi = min(len(x), i + w + 1)
        o[i] = np.std(x[lo:hi])
    return o


# ===========================================================================
# FIGURA 1 — Perfis de ativação do pulso
# ===========================================================================
profiles = ["square", "sine", "cos2", "triangle", "gaussian", "blackman", "gravity_sine"]
tau = np.linspace(0.0, 1.0, 400)
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for p in profiles:
    e = profile_envelope(p, tau, sigma=0.2, beta=0.09)
    ax.plot(tau, e, lw=1.8, label=p)
ax.set_xlabel(r"$\tau = t / t_{\mathrm{cav}}$")
ax.set_ylabel(r"$\mathrm{env}(\tau)$ (área unitária)")
ax.set_title("Perfis de ativação do campo de micro-ondas")
ax.legend(ncol=2, fontsize=8)
ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(FIG + "/fig_perfis.png", dpi=160)
plt.close(fig)

# ===========================================================================
# FIGURA 2 — Validação: simulação otimizada vs fountain_sim_alt
# ===========================================================================
from fountain_sim_alt import simulate_ramsey_fringe as sim_alt  # noqa: E402

temp, T, t, th = 1.2e-6, 0.329, 0.0178, 1.549016
tv_ref, P2_ref = sim_alt(temp, T, t, B, th, th, N_r=50, N_v=30,
                         profile1="sine", profile2="sine")
_, P2_perf = simulate_ramsey_fringe(temp, T, t, B, th, th, N_r=20, N_v=15,
                                    N_steps=16, profile1="sine", profile2="sine",
                                    tvec=tv_ref)
dv_m = tv_ref / (2 * np.pi * t)
diff = P2_perf - P2_ref

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.2, 4.2),
                               gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(dv_m, P2_ref, "-", lw=2.0, color="black", label="fountain_sim_alt (N=50×30)")
ax1.plot(dv_m, P2_perf, "--", lw=1.6, color="red", label="fountain_sim_performance (20×15)")
ax1.set_ylabel(r"$P_2$")
ax1.set_title("Validação da simulação otimizada (pulso seno)")
ax1.legend(fontsize=8)
ax2.plot(dv_m, diff, "-", color="blue", lw=1.0)
ax2.axhline(0, color="gray", lw=0.8)
ax2.set_xlabel(r"Dessintonia $\Delta\nu$ (Hz)")
ax2.set_ylabel("Diferença")
fig.tight_layout()
fig.savefig(FIG + "/fig_validacao.png", dpi=160)
plt.close(fig)

# ===========================================================================
# FIGURA 3 — "Nó" do pulso quadrado vs decaimento suave do Blackman
# ===========================================================================
t_demo = 0.0178
tv_demo = 2 * np.pi * dv * t_demo
_, P2_sq = simulate_ramsey_fringe(64.3e-6, 0.331, t_demo, B, np.pi/2, np.pi/2,
                                  N_r=20, N_v=15, N_steps=16, profile1="square",
                                  profile2="square", tvec=tv_demo)
_, P2_bm = simulate_ramsey_fringe(64.3e-6, 0.331, t_demo, B, np.pi/2, np.pi/2,
                                  N_r=20, N_v=15, N_steps=16, profile1="blackman",
                                  profile2="blackman", tvec=tv_demo)

y_norm = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
amp_data = local_amp(y_norm)
amp_sq = local_amp(P2_sq)
amp_bm = local_amp(P2_bm)

fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(dv, amp_data, "-", color="black", lw=1.6, label="Experimental")
ax.plot(dv, amp_sq, "-", color="blue", lw=1.6, label="Pulso quadrado")
ax.plot(dv, amp_bm, "-", color="red", lw=1.6, label="Pulso Blackman")
ax.axvline(56.2, color="blue", ls=":", lw=1.0)
ax.text(57, 0.32, "nó do sinc\n(≈56 Hz)", color="blue", fontsize=8)
ax.set_xlabel(r"Dessintonia $\Delta\nu$ (Hz)")
ax.set_ylabel("Amplitude local da franja (std)")
ax.set_title("Artefato do pulso quadrado (nó de sinc) vs Blackman")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG + "/fig_no_sinc.png", dpi=160)
plt.close(fig)

print("Figuras 1-3 geradas.")



# ===========================================================================
# FIGURA 4 — Antes (min-max) vs Depois (resposta linear)
# ===========================================================================
def linear_response(P2, y):
    pm, ym = P2.mean(), y.mean()
    den = np.sum((P2 - pm) ** 2)
    A = np.sum((P2 - pm) * (y - ym)) / den if den > 1e-15 else 0.0
    return A, ym - A * pm


def run(profile, p, tvec):
    return simulate_ramsey_fringe(p[0] * 1e-6, p[1], p[2], B, np.pi/2, np.pi/2,
                                  N_r=20, N_v=15, N_steps=16,
                                  profile1=profile, profile2=profile, tvec=tvec)[1]


y_norm = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())
tvec_exp_old = 2 * np.pi * dv * 0.036463

# Antes: min-max (ajuste simétrico Blackman antigo, θ≈π/2)
P2_old = simulate_ramsey_fringe(0.855805e-6, 0.310027, 0.036463, B,
                                1.571657, 1.571657, N_r=20, N_v=15, N_steps=16,
                                profile1="blackman", profile2="blackman",
                                tvec=tvec_exp_old)[1]
r_old = P2_old - y_norm
r2_old = 1 - np.sum(r_old**2) / np.sum((y_norm - y_norm.mean())**2)

# Depois: resposta linear (melhor perfil: square)
p_new = [64.341386, 0.331055, 0.017580]
P2_new = run("square", p_new, 2 * np.pi * dv * p_new[2])
A_new, C_new = linear_response(P2_new, y_raw)
r_new = (A_new * P2_new + C_new) - y_raw
r2_new = 1 - np.sum(r_new**2) / np.sum((y_raw - y_raw.mean())**2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3.4))
ax1.bar(["Min-max", "Resposta linear"], [r2_old, r2_new],
        color=["#999999", "#d62728"], width=0.55)
ax1.set_ylabel(r"$R^2$")
ax1.set_ylim(0, 1.05)
for i, v in enumerate([r2_old, r2_new]):
    ax1.text(i, v + 0.02, f"{v:.3f}", ha="center")
ax1.set_title("Qualidade global do ajuste")

m = (np.abs(dv) >= 15) & (np.abs(dv) <= 45)
rmse_old = np.sqrt(np.mean(r_old[m]**2))
rmse_new = np.sqrt(np.mean(r_new[m]**2))
ax2.bar(["Min-max", "Resposta linear"], [rmse_old, rmse_new],
        color=["#999999", "#d62728"], width=0.55)
ax2.set_ylabel("RMSE (região 15–45 Hz)")
for i, v in enumerate([rmse_old, rmse_new]):
    ax2.text(i, v + 0.001, f"{v:.4f}", ha="center")
ax2.set_title("Resíduo nos vales (washing-out)")
fig.tight_layout()
fig.savefig(FIG + "/fig_antes_depois.png", dpi=160)
plt.close(fig)

# ===========================================================================
# FIGURA 5 — Ajuste final (melhor perfil: square) + resíduos
# ===========================================================================
dv_fine = np.linspace(dv.min(), dv.max(), 2000)
P2_fine = run("square", p_new, 2 * np.pi * dv_fine * p_new[2])
y_fine = A_new * P2_fine + C_new
P2_exp = run("square", p_new, 2 * np.pi * dv * p_new[2])
res = (A_new * P2_exp + C_new) - y_raw

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6.4, 4.6),
                               gridspec_kw={"height_ratios": [3, 1]})
ax1.plot(dv, y_raw, "o", ms=3, color="black", alpha=0.5, label="Experimental")
ax1.plot(dv_fine, y_fine, "-", color="red", lw=2.0, label="Ajuste (square, y=A·P2+C)")
ax1.set_ylabel("Sinal bruto")
ax1.set_title("Ajuste final das franjas de Ramsey")
ax1.legend(fontsize=8)
ax2.plot(dv, res, "o", ms=3, color="blue", alpha=0.6)
ax2.axhline(0, color="gray", lw=0.8)
ax2.set_xlabel(r"Dessintonia $\Delta\nu$ (Hz)")
ax2.set_ylabel("Resíduo")
fig.tight_layout()
fig.savefig(FIG + "/fig_ajuste_final.png", dpi=160)
plt.close(fig)

# ===========================================================================
# FIGURA 6 — Seleção de modelo (R², RMSE, AIC por perfil)
# ===========================================================================
best_params = {
    "square":      [64.341386, 0.331055, 0.017580],
    "sine":        [72.620691, 0.325053, 0.023610],
    "cos2":        [80.882032, 0.319961, 0.028738],
    "triangle":    [76.027246, 0.323354, 0.025330],
    "gaussian":    [80.094932, 0.321301, 0.027406],
    "blackman":    [88.547912, 0.315992, 0.032742],
    "gravity_sine":[72.653938, 0.325046, 0.023617],
}
rows = []
for prof, p in best_params.items():
    P2 = run(prof, p, 2 * np.pi * dv * p[2])
    A, C = linear_response(P2, y_raw)
    r = (A * P2 + C) - y_raw
    ss_res = np.sum(r**2)
    ss_tot = np.sum((y_raw - y_raw.mean())**2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(r**2))
    aic = len(y_raw) * np.log(ss_res / len(y_raw)) + 2 * (len(p) + 2)
    rows.append((prof, r2, rmse, aic))

profiles = [r[0] for r in rows]
r2s = [r[1] for r in rows]
rmses = [r[2] for r in rows]
aics = [r[3] for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
x = np.arange(len(profiles))
axes[0].bar(x, r2s, color="#1f77b4")
axes[0].set_xticks(x); axes[0].set_xticklabels(profiles, rotation=45, fontsize=8)
axes[0].set_ylabel(r"$R^2$"); axes[0].set_ylim(0.97, 0.985)
axes[0].set_title(r"$R^2$ (maior é melhor)")
axes[1].bar(x, rmses, color="#2ca02c")
axes[1].set_xticks(x); axes[1].set_xticklabels(profiles, rotation=45, fontsize=8)
axes[1].set_ylabel("RMSE"); axes[1].set_title("RMSE (menor é melhor)")
axes[2].bar(x, aics, color="#ff7f0e")
axes[2].set_xticks(x); axes[2].set_xticklabels(profiles, rotation=45, fontsize=8)
axes[2].set_ylabel("AIC"); axes[2].set_title("AIC (menor é melhor)")
fig.tight_layout()
fig.savefig(FIG + "/fig_selecao_modelo.png", dpi=160)
plt.close(fig)

print("\nNúmeros da seleção de modelo:")
for prof, r2, rmse, aic in rows:
    print(f"  {prof:<14} R2={r2:.6f}  RMSE={rmse:.6f}  AIC={aic:.2f}")
print(f"\nA/C (square): A={A_new:.6f}  C={C_new:.6f}")
print(f"R2 antes={r2_old:.6f}  depois={r2_new:.6f}")

print("\nTodas as figuras geradas.")
