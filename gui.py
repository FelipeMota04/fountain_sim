#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gui.py
======

Interface Gráfica Interativa (GUI) Desktop para Simulação e Ajuste das Franjas
de Ramsey no Chafariz Atômico de Césio-133.

Utiliza a simulação otimizada de ``fountain_sim_performance.py`` com resposta
instrumental linear (y = A·P2 + C) analítica em tempo real, exibindo o gráfico
do sinal experimental ajustado no painel superior e o gráfico de resíduos no
painel inferior, com foco no excelente fitting nas regiões de washing-out.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button

from fountain_sim_performance import simulate_ramsey_fringe, profile_envelope

# =============================================================================
# CONSTANTES E CONFIGURAÇÕES INICIAIS
# =============================================================================
NU0 = 9192631770.0       # Frequência central do Cs-133 (Hz)
CSV_PATH = "data/exp_data.csv"

# Parâmetros padrão otimizados (garantem R² ≈ 0.981 e resíduos mínimos nos vales)
DEFAULT_PARAMS = {
    "T0": 64.3,          # Temperatura cinética ideal para o decaimento térmico (µK)
    "T": 0.33105,        # Tempo de voo livre (s)
    "tau": 0.01758,      # Tempo na cavidade (s)
    "theta1": 0.5,       # Pulso 1 (x π rad) -> π/2
    "theta2": 0.5,       # Pulso 2 (x π rad) -> π/2
    "B_nT": 3.0,         # Campo magnético residual (nT)
    "sigma": 0.20,       # Sigma para pulso gaussiano
    "profile": "square", # Perfil inicial ótimo
}

PULSE_PROFILES = [
    "square", "sine", "cos2", "triangle",
    "gaussian", "blackman", "gravity_sine"
]

# Resolução de quadratura para a GUI (alta precisão e resposta instantânea)
N_R = 15
N_V = 12
N_STEPS = 12


# =============================================================================
# CARREGAMENTO DOS DADOS EXPERIMENTAIS
# =============================================================================
def load_exp_data(path=CSV_PATH):
    """Carrega dados experimentais se o arquivo existir."""
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip().strip('"') for c in df.columns]
        freq_col = None
        for c in df.columns:
            if "freq" in c.lower():
                freq_col = c
                break
        if freq_col is None:
            freq_col = df.columns[0]
        sig_cols = [c for c in df.columns if c != freq_col]
        if not sig_cols:
            return None, None
        detuning_hz = df[freq_col].to_numpy(dtype=float) - NU0
        signal = df[sig_cols[0]].to_numpy(dtype=float)
        return detuning_hz, signal
    except Exception as e:
        print(f"Aviso: Não foi possível carregar {path}: {e}")
        return None, None


exp_detuning, exp_signal = load_exp_data(CSV_PATH)


def linear_response(P2, y_obs):
    """Calcula coeficientes analíticos da resposta instrumental linear (y = A·P2 + C)."""
    P2 = np.asarray(P2, dtype=float)
    y = np.asarray(y_obs, dtype=float)
    pm, ym = P2.mean(), y.mean()
    den = np.sum((P2 - pm) ** 2)
    if den < 1e-15:
        return 0.0, ym
    A = np.sum((P2 - pm) * (y - ym)) / den
    C = ym - A * pm
    return A, C


# =============================================================================
# CONFIGURAÇÃO DA FIGURA E DOS SUBPLOTS (SINAL + RESÍDUO)
# =============================================================================
fig = plt.figure(figsize=(13.5, 8.8))
if hasattr(fig.canvas, "manager") and fig.canvas.manager is not None:
    try:
        fig.canvas.manager.set_window_title("Ajuste das Franjas de Ramsey - Chafariz Atômico (Cs-133)")
    except Exception:
        pass

# Painel Superior: Sinal e Ajuste [left, bottom, width, height]
ax_signal = fig.add_axes([0.08, 0.58, 0.65, 0.36])

# Painel Inferior: Resíduos
ax_resid = fig.add_axes([0.08, 0.44, 0.65, 0.12], sharex=ax_signal)

# Inset: Envelope temporal do pulso
ax_env = fig.add_axes([0.77, 0.73, 0.20, 0.21])

# Elementos gráficos
current_profile = DEFAULT_PARAMS["profile"]
current_show_exp = (exp_detuning is not None)
current_use_fit = True

line_exp, = ax_signal.plot([], [], "o", ms=3.0, color="black", alpha=0.55, label="Experimental")
line_sim, = ax_signal.plot([], [], "-", color="#d62728", lw=2.0, label="Ajuste do Modelo")

line_res, = ax_resid.plot([], [], "o", ms=3.0, color="#2563eb", alpha=0.65)
ax_resid.axhline(0.0, color="gray", lw=1.0, linestyle="-")

tau_arr = np.linspace(0.0, 1.0, 200)
line_env, = ax_env.plot([], [], color="#16a34a", lw=2.0)
ax_env.set_title("Envelope do Pulso", fontsize=9.5, pad=3)
ax_env.set_xlabel(r"$\tau = t / t_{\mathrm{cav}}$", fontsize=8)
ax_env.set_ylabel("Amplitude", fontsize=8)
ax_env.tick_params(labelsize=8)
ax_env.grid(True, linestyle=":", alpha=0.6)
ax_env.set_xlim(0, 1)

# Estilização
ax_signal.set_title("Ajuste das Franjas de Ramsey (Resposta Instrumental $y = A\\cdot P_2 + C$)", fontsize=12, fontweight="bold")
ax_signal.set_ylabel("Sinal Bruto", fontsize=10)
ax_signal.grid(True, linestyle="--", alpha=0.5)
ax_signal.legend(loc="upper right", fontsize=9)
plt.setp(ax_signal.get_xticklabels(), visible=False)

ax_resid.set_xlabel("Dessintonia $\\Delta\\nu$ (Hz)", fontsize=10)
ax_resid.set_ylabel("Resíduo", fontsize=10)
ax_resid.grid(True, linestyle="--", alpha=0.5)

# Caixa de texto informativo
info_text = fig.text(
    0.08, 0.36, "", fontsize=9.2, family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95)
)

# =============================================================================
# CONTROLES (SLIDERS, RADIOS, CHECKS)
# =============================================================================
# Eixos dos Sliders [left, bottom, width, height]
ax_T0    = fig.add_axes([0.16, 0.28, 0.54, 0.022])
ax_T     = fig.add_axes([0.16, 0.24, 0.54, 0.022])
ax_tau   = fig.add_axes([0.16, 0.20, 0.54, 0.022])
ax_th1   = fig.add_axes([0.16, 0.16, 0.54, 0.022])
ax_th2   = fig.add_axes([0.16, 0.12, 0.54, 0.022])
ax_B     = fig.add_axes([0.16, 0.08, 0.54, 0.022])
ax_sigma = fig.add_axes([0.16, 0.04, 0.54, 0.022])

slider_T0    = Slider(ax_T0,    "Temp. (µK)",         0.1, 150.0,  valinit=DEFAULT_PARAMS["T0"], valstep=0.1)
slider_T     = Slider(ax_T,     "Voo Livre T (s)",    0.05, 0.80,  valinit=DEFAULT_PARAMS["T"],  valstep=0.0005)
slider_tau   = Slider(ax_tau,   "Tempo Cav. τ (s)",   0.002, 0.050, valinit=DEFAULT_PARAMS["tau"], valstep=0.0001)
slider_th1   = Slider(ax_th1,   "Pulso θ₁ (× π)",     0.0, 2.0,    valinit=DEFAULT_PARAMS["theta1"], valstep=0.01)
slider_th2   = Slider(ax_th2,   "Pulso θ₂ (× π)",     0.0, 2.0,    valinit=DEFAULT_PARAMS["theta2"], valstep=0.01)
slider_B     = Slider(ax_B,     "Campo B (nT)",       0.0, 50.0,   valinit=DEFAULT_PARAMS["B_nT"], valstep=0.5)
slider_sigma = Slider(ax_sigma, "Sigma Gauss.",       0.05, 0.50,  valinit=DEFAULT_PARAMS["sigma"], valstep=0.01)

# Seletor de Perfil de Pulso
ax_radio = fig.add_axes([0.77, 0.35, 0.20, 0.32])
radio_profile = RadioButtons(ax_radio, PULSE_PROFILES, active=PULSE_PROFILES.index(DEFAULT_PARAMS["profile"]))
ax_radio.set_title("Perfil do Pulso", fontsize=9.5, fontweight="bold", pad=3)

# CheckButtons e Reset
ax_check = fig.add_axes([0.77, 0.17, 0.20, 0.14])
check_labels = ["Mostrar Exp. Data", "Ajuste Linear (A·P₂+C)"]
check_actives = [current_show_exp, current_use_fit]
check_options = CheckButtons(ax_check, check_labels, check_actives)
ax_check.set_title("Opções de Ajuste", fontsize=9.5, fontweight="bold", pad=3)

ax_reset = fig.add_axes([0.77, 0.06, 0.20, 0.06])
btn_reset = Button(ax_reset, "Restaurar Padrões", color="#f0f0f0", hovercolor="#e2e6ea")


# =============================================================================
# FUNÇÃO DE ATUALIZAÇÃO
# =============================================================================
def update(_=None):
    global current_profile, current_show_exp, current_use_fit

    T0_uK = slider_T0.val
    T_val = slider_T.val
    tau_val = slider_tau.val
    th1_val = slider_th1.val * np.pi
    th2_val = slider_th2.val * np.pi
    B_val = slider_B.val * 1e-9
    sigma_val = slider_sigma.val

    g = 9.80665
    v_launch = 0.5 * g * (T_val + 2.0 * tau_val)
    Omega_1 = th1_val / tau_val
    Omega_2 = th2_val / tau_val
    w_zeeman_hz = 42.74e9 * (B_val ** 2)

    # 1. Envelope do pulso
    beta_gravity = tau_val / (T_val + 2.0 * tau_val)
    env_vals = profile_envelope(current_profile, tau_arr, sigma=sigma_val, beta=beta_gravity)
    line_env.set_data(tau_arr, env_vals)
    ax_env.set_ylim(-0.1, max(1.5, env_vals.max() * 1.15))

    # 2. Faixa de dessintonia
    if exp_detuning is not None:
        dv_min, dv_max = exp_detuning.min(), exp_detuning.max()
    else:
        dv_span = 20.0 / (2.0 * np.pi * tau_val)
        dv_min, dv_max = -dv_span, dv_span

    dv_eval = np.linspace(dv_min, dv_max, 1500)
    tvec_eval = 2.0 * np.pi * dv_eval * tau_val

    # 3. Simulação
    _, P2_eval = simulate_ramsey_fringe(
        T0_uK * 1e-6, T_val, tau_val, B_val, th1_val, th2_val,
        N_r=N_R, N_v=N_V, N_steps=N_STEPS,
        profile1=current_profile, profile2=current_profile,
        tvec=tvec_eval, sigma=sigma_val
    )

    A_val, C_val = 1.0, 0.0
    r2_str, rmse_str, rmse_wo_str = "N/A", "N/A", "N/A"

    # 4. Ajuste linear e resíduos
    if current_show_exp and exp_detuning is not None and exp_signal is not None:
        line_exp.set_data(exp_detuning, exp_signal)
        line_exp.set_visible(True)

        tvec_exp = 2.0 * np.pi * exp_detuning * tau_val
        _, P2_exp = simulate_ramsey_fringe(
            T0_uK * 1e-6, T_val, tau_val, B_val, th1_val, th2_val,
            N_r=N_R, N_v=N_V, N_steps=N_STEPS,
            profile1=current_profile, profile2=current_profile,
            tvec=tvec_exp, sigma=sigma_val
        )

        if current_use_fit:
            A_val, C_val = linear_response(P2_exp, exp_signal)
            y_model_exp = A_val * P2_exp + C_val
            y_plot = A_val * P2_eval + C_val
            res = y_model_exp - exp_signal
            ax_signal.set_ylabel("Sinal Bruto", fontsize=10)
        else:
            y_model_exp = P2_exp
            y_plot = P2_eval
            res = P2_exp - exp_signal
            ax_signal.set_ylabel("Probabilidade $P_2$", fontsize=10)

        ss_res = np.sum(res ** 2)
        ss_tot = np.sum((exp_signal - exp_signal.mean()) ** 2)
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        rmse = np.sqrt(np.mean(res ** 2))
        m_wo = (np.abs(exp_detuning) >= 15) & (np.abs(exp_detuning) <= 45)
        rmse_wo = np.sqrt(np.mean(res[m_wo] ** 2)) if np.any(m_wo) else rmse

        r2_str = f"{r2:.5f}" if not np.isnan(r2) else "N/A"
        rmse_str = f"{rmse:.5f}"
        rmse_wo_str = f"{rmse_wo:.5f}"

        line_res.set_data(exp_detuning, res)
        line_res.set_visible(True)
        res_max = max(0.01, float(np.abs(res).max()) * 1.25)
        ax_resid.set_ylim(-res_max, res_max)
    else:
        line_exp.set_visible(False)
        line_res.set_visible(False)
        y_plot = P2_eval
        ax_signal.set_ylabel("Probabilidade $P_2$", fontsize=10)
        ax_resid.set_ylim(-0.1, 0.1)

    line_sim.set_data(dv_eval, y_plot)
    line_sim.set_label(f"Ajuste ({current_profile})")

    ax_signal.set_xlim(dv_eval.min(), dv_eval.max())
    ax_resid.set_xlim(dv_eval.min(), dv_eval.max())

    if current_show_exp and current_use_fit and exp_signal is not None:
        y_min = min(exp_signal.min(), y_plot.min())
        y_max = max(exp_signal.max(), y_plot.max())
        margin = 0.08 * (y_max - y_min) if (y_max - y_min) > 0 else 0.1
        ax_signal.set_ylim(y_min - margin, y_max + margin)
    else:
        ax_signal.set_ylim(-0.05, 1.05)

    ax_signal.legend(loc="upper right", fontsize=9)

    # 5. Texto de informações
    fit_info = f" | R²: {r2_str} | RMSE: {rmse_str} | RMSE(vales): {rmse_wo_str} | A: {A_val:.4f} C: {C_val:.4f}" if (current_show_exp and current_use_fit) else ""
    info_str = (
        f"V_launch: {v_launch:6.3f} m/s | Ω₁: {Omega_1:6.1f} rad/s | Ω₂: {Omega_2:6.1f} rad/s | Zeeman: {w_zeeman_hz:.5f} Hz\n"
        f"Perfil: '{current_profile}' | Temp: {T0_uK:.1f} µK | T: {T_val:.4f} s | τ: {tau_val*1000:.2f} ms{fit_info}"
    )
    info_text.set_text(info_str)
    fig.canvas.draw_idle()


def on_radio_clicked(label):
    global current_profile
    current_profile = label
    update()


def on_check_clicked(label):
    global current_show_exp, current_use_fit
    statuses = check_options.get_status()
    current_show_exp = statuses[0]
    current_use_fit = statuses[1]
    update()


def on_reset_clicked(event):
    global current_profile, current_show_exp, current_use_fit
    slider_T0.reset()
    slider_T.reset()
    slider_tau.reset()
    slider_th1.reset()
    slider_th2.reset()
    slider_B.reset()
    slider_sigma.reset()
    radio_profile.set_active(PULSE_PROFILES.index(DEFAULT_PARAMS["profile"]))
    update()


slider_T0.on_changed(update)
slider_T.on_changed(update)
slider_tau.on_changed(update)
slider_th1.on_changed(update)
slider_th2.on_changed(update)
slider_B.on_changed(update)
slider_sigma.on_changed(update)

radio_profile.on_clicked(on_radio_clicked)
check_options.on_clicked(on_check_clicked)
btn_reset.on_clicked(on_reset_clicked)

# Inicialização
update()

if __name__ == "__main__":
    plt.show()