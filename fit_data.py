#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fit_data.py
===========

Ajusta as franjas de Ramsey (pulso NÃO-instantâneo, otimizado em
``fountain_sim_performance.py``) aos dados experimentais de ``exp_data.csv``.

Nesta versão:
  * o sinal experimental é modelado com resposta instrumental LINEAR:
        y_obs = A·P2(Δν; físicos) + C
    onde A (contraste/escala) e C (fundo/offset) são "perfilados" analiticamente
    (mínimos quadrados lineares), evitando a distorção da normalização min-max;
  * a área do pulso é fixada em θ = π/2 (físico para um experimento de Ramsey);
  * vários perfis de ativação do micro-ondas são testados e comparados
    (seleção de modelo), escolhendo o que melhor reproduz os dados.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # modo headless; remova esta linha para exibir a janela
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from fountain_sim_performance import simulate_ramsey_fringe

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
CSV_PATH = "data/exp_data.csv"     # arquivo com os dados experimentais
NU0 = 9192631770.0                 # frequência central do Cs-133 (Hz)
B_FIXED = 3.0e-9                   # campo magnético fixo (T) -> 3 nT
OUTPUT_PLOT = "images/fit_ramsey.png"  # arquivo do gráfico gerado

# Perfis de ativação do micro-ondas testados (seleção de modelo).
PULSE_PROFILES = ["square", "sine", "cos2", "triangle", "gaussian",
                  "blackman", "gravity_sine"]

# Largura do perfil gaussiano (fixa, não ajustada).
GAUSSIAN_SIGMA = 0.2

# Geometria experimental (MOT e aberturas da cavidade)
H_MOT_FIXED = 0.0         # distância MOT -> cavidade (m). Ex.: 0.35 para chafariz com MOT inferior
SIGMA_R0_FIXED = 0.0      # raio inicial da nuvem (m). Ex.: 1.5e-3 (1.5 mm)
R_APERTURE_FIXED = None   # raio da abertura da cavidade (m). Ex.: 0.010 (10 mm) ou None (Rc=21.5 mm)

# Fixa a área do pulso em θ = π/2 (físico para Ramsey).
FIX_THETA_PI2 = True

# Resoluções da simulação otimizada (fountain_sim_performance):
#   N_R      -> pontos de Gauss-Legendre (integral radial, [0, Rc])
#   N_V      -> pontos de Gauss-Hermite (integral de velocidade)
#   N_STEPS  -> passos de integração do envelope do pulso
N_R_FIT, N_V_FIT, N_STEPS_FIT = 10, 8, 8
N_R_REFINE, N_V_REFINE, N_STEPS_REFINE = 15, 12, 12
N_R_FINAL, N_V_FINAL, N_STEPS_FINAL = 20, 15, 16

# Parâmetros físicos ajustáveis: [Temperatura (µK), T (s), tau (s)]
# (θ = π/2 é fixo quando FIX_THETA_PI2 = True).
if FIX_THETA_PI2:
    LOWER = [0.05, 0.05, 0.001]
    UPPER = [500.0, 0.80, 0.050]
else:
    LOWER = [0.05, 0.05, 0.001, 0.05]
    UPPER = [500.0, 0.80, 0.050, np.pi]

MAX_NFEV_COARSE = 80
MAX_NFEV_REFINE = 60


# =============================================================================
# 1. LEITURA DOS DADOS EXPERIMENTAIS
# =============================================================================
def load_experimental_data(csv_path):
    """Lê o CSV e devolve {coluna_sinal: (detuning_hz, sinal_bruto)}."""
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().strip('"').strip() for c in df.columns]

    freq_col = None
    for c in df.columns:
        if "freq" in c.lower():
            freq_col = c
            break
    if freq_col is None:
        freq_col = df.columns[0]

    signal_cols = [c for c in df.columns if c != freq_col]
    if not signal_cols:
        raise ValueError("Nenhuma coluna de sinal encontrada no CSV.")

    freq = df[freq_col].to_numpy(dtype=float)
    detuning_hz = freq - NU0

    results = {}
    for col in signal_cols:
        results[col] = (detuning_hz, df[col].to_numpy(dtype=float))

    print(f"CSV lido: '{csv_path}'")
    print(f"  Coluna de frequência : '{freq_col}'")
    print(f"  Colunas de sinal     : {signal_cols}")
    print(f"  Faixa de dessintonia : {detuning_hz.min():.3f} a "
          f"{detuning_hz.max():.3f} Hz ({len(detuning_hz)} pontos)")
    print("  Resposta instrumental: y_obs = A·P2 + C (A e C ajustados por mín. quadrados)")
    return results


# =============================================================================
# 2. MODELO, RESPOSTA INSTRUMENTAL E RESÍDUOS
# =============================================================================
def estimate_free_flight_time(detuning_hz, signal):
    """Estima T (s) pela frequência dominante da franja (FFT)."""
    y = signal - np.mean(signal)
    d = np.median(np.diff(detuning_hz))
    freqs = np.fft.rfftfreq(len(y), d=d)
    spec = np.abs(np.fft.rfft(y))
    mask = freqs > 0.05
    if not np.any(mask):
        return 0.35
    peak = freqs[mask][np.argmax(spec[mask])]
    return float(np.clip(peak, LOWER[1], UPPER[1]))


def unpack_params(p):
    """Converte o vetor de parâmetros em (temp_uK, T, tau, theta1, theta2)."""
    if FIX_THETA_PI2:
        temp_uK, T, tau = p
        theta = np.pi / 2
        return temp_uK, T, tau, theta, theta
    temp_uK, T, tau, theta = p
    return temp_uK, T, tau, theta, theta


def run_simulation(temp_uK, T, tau, theta1, theta2, tvec, n_r, n_v, n_steps, profile):
    """Chama a simulação otimizada com o perfil de pulso escolhido."""
    return simulate_ramsey_fringe(
        temp_uK * 1e-6, T, tau, B_FIXED, theta1, theta2,
        N_r=n_r, N_v=n_v, N_steps=n_steps,
        profile1=profile, profile2=profile,
        tvec=tvec, sigma=GAUSSIAN_SIGMA,
        h_mot=H_MOT_FIXED, sigma_r0=SIGMA_R0_FIXED,
        R_aperture=R_APERTURE_FIXED
    )


def linear_response(P2, y_raw):
    """Ajuste linear y_raw = A*P2 + C; devolve (A, C)."""
    P2 = np.asarray(P2, dtype=float)
    y = np.asarray(y_raw, dtype=float)
    pm = P2.mean()
    ym = y.mean()
    den = np.sum((P2 - pm) ** 2)
    if den < 1e-15:
        return 0.0, ym
    A = np.sum((P2 - pm) * (y - ym)) / den
    C = ym - A * pm
    return A, C


def residuals(p, detuning_hz, y_raw, n_r, n_v, n_steps, profile):
    """Resíduos: (A·P2 + C) - y_obs, com A e C perfilados analiticamente."""
    temp_uK, T, tau, theta1, theta2 = unpack_params(p)
    tvec_exp = 2.0 * np.pi * detuning_hz * tau
    _, P2 = run_simulation(temp_uK, T, tau, theta1, theta2,
                           tvec_exp, n_r, n_v, n_steps, profile)
    A, C = linear_response(P2, y_raw)
    return (A * P2 + C) - y_raw


# =============================================================================
# 3. AJUSTE
# =============================================================================
def build_initial_guesses(T_guess):
    """Pontos de partida (multistart) em torno de T estimado."""
    if FIX_THETA_PI2:
        return [
            [16.0, T_guess, 0.025],
            [16.0, T_guess, 0.017],
            [16.0, T_guess, 0.010],
            [5.0, T_guess, 0.025],
            [40.0, T_guess, 0.025],
            [16.0, T_guess * 0.97, 0.025],
            [16.0, T_guess * 1.03, 0.025],
        ]
    return [
        [16.0, T_guess, 0.025, np.pi / 2],
        [16.0, T_guess, 0.017, np.pi / 2],
        [5.0, T_guess, 0.025, np.pi / 2],
        [40.0, T_guess, 0.025, np.pi / 2],
    ]


def fit_curve(detuning_hz, y_raw, profile):
    """Ajuste em duas fases (multistart grosseiro + refinamento)."""
    T_guess = estimate_free_flight_time(detuning_hz, y_raw)
    best = None
    for p0 in build_initial_guesses(T_guess):
        try:
            res = least_squares(
                residuals, p0,
                args=(detuning_hz, y_raw, N_R_FIT, N_V_FIT, N_STEPS_FIT, profile),
                bounds=(LOWER, UPPER), method="trf", max_nfev=MAX_NFEV_COARSE,
            )
            cost = float(2.0 * res.cost)
            if best is None or cost < best["cost"]:
                best = {"res": res, "cost": cost, "p0": p0}
        except Exception as exc:  # noqa: BLE001
            print(f"  [aviso] chute {p0} falhou: {exc}")

    if best is None:
        raise RuntimeError("O ajuste não convergiu para nenhum ponto de partida.")

    res_refined = least_squares(
        residuals, best["res"].x,
        args=(detuning_hz, y_raw, N_R_REFINE, N_V_REFINE, N_STEPS_REFINE, profile),
        bounds=(LOWER, UPPER), method="trf", max_nfev=MAX_NFEV_REFINE,
    )
    cost_refined = float(2.0 * res_refined.cost)
    if cost_refined < best["cost"]:
        best = {"res": res_refined, "cost": cost_refined, "p0": best["p0"]}
    return best, T_guess


def parameter_errors(res):
    """Desvios-padrão dos parâmetros via matriz Jacobiana."""
    J = res.jac
    if J is None:
        return np.full(len(res.x), np.nan)
    cost = float(2.0 * res.cost)
    dof = len(res.fun) - len(res.x)
    s2 = cost / dof if dof > 1 else 0.0
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        return np.sqrt(np.diag(np.clip(cov, 0, None)))
    except np.linalg.LinAlgError:
        return np.full(len(res.x), np.nan)


def evaluate(p, detuning_hz, y_raw, profile):
    """Calcula A, C e métricas de qualidade (R², RMSE, AIC)."""
    temp_uK, T, tau, theta1, theta2 = unpack_params(p)
    tvec_exp = 2.0 * np.pi * detuning_hz * tau
    _, P2 = run_simulation(temp_uK, T, tau, theta1, theta2,
                           tvec_exp, N_R_REFINE, N_V_REFINE, N_STEPS_REFINE, profile)
    A, C = linear_response(P2, y_raw)
    y_fit = A * P2 + C
    res = y_fit - y_raw
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((y_raw - y_raw.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(res ** 2)))
    n = len(y_raw)
    k = len(p) + 2  # parâmetros físicos + A + C
    aic = n * np.log(ss_res / n) + 2.0 * k if ss_res > 0 else -np.inf
    return A, C, r2, rmse, aic


# =============================================================================
# 4. RESULTADOS
# =============================================================================
def report_profile(profile, p, err, A, C, r2, rmse, aic, detuning_hz, y_raw):
    temp_uK, T, tau, theta1, theta2 = unpack_params(p)
    g = 9.80665
    Omega = theta1 / tau
    v_launch = 0.5 * g * (T + 2.0 * tau)
    w_zeeman_hz = 42.74e9 * B_FIXED ** 2

    names = ["Temperatura (µK)", "Tempo de voo livre T (s)", "Tempo na cavidade τ (s)"]
    print("\n" + "=" * 66)
    print(f"PERFIL '{profile}'  (θ = π/2 fixo)")
    print("=" * 66)
    for name, val, e in zip(names, p, err):
        if np.isnan(e):
            print(f"  {name:<26}: {val:12.6f}")
        else:
            print(f"  {name:<26}: {val:12.6f} ± {e:.6f}")
    print(f"  {'Pulso θ (rad, fixo)':<26}: {theta1:12.6f}  (= {theta1 / np.pi:.4f} π)")
    print("-" * 66)
    print("Resposta instrumental (y = A·P2 + C):")
    print(f"  Contraste/escala A        : {A:12.6f}")
    print(f"  Fundo/offset C            : {C:12.6f}")
    print("-" * 66)
    print("Grandezas derivadas:")
    print(f"  Frequência de Rabi Ω = θ/τ : {Omega:.4f} rad/s ({Omega / (2 * np.pi):.4f} Hz)")
    print(f"  Velocidade de lançamento  : {v_launch:.4f} m/s")
    print(f"  Campo magnético B (fixo)  : {B_FIXED * 1e9:.2f} nT")
    print(f"  Efeito Zeeman (2ª ordem)  : {w_zeeman_hz:.6f} Hz")
    print("-" * 66)
    print("Qualidade do ajuste:")
    print(f"  R²                        : {r2:.6f}")
    print(f"  RMSE                      : {rmse:.6f}")
    print(f"  AIC                       : {aic:.2f}")
    print("=" * 66)
    return Omega



# =============================================================================
# 5. GRÁFICO
# =============================================================================
def make_plot(detuning_hz, y_raw, p, profile, A, C, col_name, output_path):
    temp_uK, T, tau, theta1, theta2 = unpack_params(p)

    dv_fine = np.linspace(detuning_hz.min(), detuning_hz.max(), 2000)
    tvec_fine = 2.0 * np.pi * dv_fine * tau
    _, P2_fine = run_simulation(temp_uK, T, tau, theta1, theta2,
                                tvec_fine, N_R_FINAL, N_V_FINAL, N_STEPS_FINAL, profile)
    y_fine = A * P2_fine + C

    tvec_exp = 2.0 * np.pi * detuning_hz * tau
    _, P2_exp = run_simulation(temp_uK, T, tau, theta1, theta2,
                               tvec_exp, N_R_FINAL, N_V_FINAL, N_STEPS_FINAL, profile)
    res = (A * P2_exp + C) - y_raw

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 7),
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(detuning_hz, y_raw, "o", ms=3, color="black", alpha=0.55,
             label=f"Experimental — {col_name}")
    ax1.plot(dv_fine, y_fine, "-", color="red", lw=2.0,
             label=f"Ajuste (perfil '{profile}')")
    ax1.set_ylabel("Sinal bruto")
    ax1.set_title("Ajuste das franjas de Ramsey")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right")

    ax2.plot(detuning_hz, res, "o", ms=3, color="blue", alpha=0.6)
    ax2.axhline(0.0, color="gray", lw=1.0)
    ax2.set_xlabel("Dessintonia $\\Delta\\nu$ (Hz)")
    ax2.set_ylabel("Resíduo")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nGráfico salvo em: '{output_path}'")


# =============================================================================
# 6. MAIN — seleção de modelo
# =============================================================================
def main():
    data = load_experimental_data(CSV_PATH)

    for col_name, (detuning_hz, y_raw) in data.items():
        print(f"\n--- Ajustando a curva '{col_name}' ---")
        results = []
        for profile in PULSE_PROFILES:
            try:
                best, _ = fit_curve(detuning_hz, y_raw, profile)
                p = best["res"].x
                err = parameter_errors(best["res"])
                A, C, r2, rmse, aic = evaluate(p, detuning_hz, y_raw, profile)
                results.append((profile, p, err, A, C, r2, rmse, aic))
                report_profile(profile, p, err, A, C, r2, rmse, aic,
                               detuning_hz, y_raw)
            except Exception as exc:  # noqa: BLE001
                print(f"\n[erro] perfil '{profile}': {exc}")

        if not results:
            print("\nNenhum perfil convergiu.")
            return

        # Seleciona o melhor perfil por R².
        best_row = max(results, key=lambda r: r[5])
        profile, p, err, A, C, r2, rmse, aic = best_row
        print("\n" + "=" * 66)
        print("RESULTADO DA SELEÇÃO DE MODELO (melhor R²)")
        print("=" * 66)
        for prof, _, _, _, _, r2i, rmsei, aici in results:
            mark = " <-- melhor" if prof == profile else ""
            print(f"  {prof:<14} R²={r2i:.4f}  RMSE={rmsei:.5f}  AIC={aici:.1f}{mark}")
        print("=" * 66)

        suffix = f"_{col_name.strip().replace(' ', '_')}" if len(data) > 1 else ""
        out = OUTPUT_PLOT.replace(".png", f"{suffix}.png")
        make_plot(detuning_hz, y_raw, p, profile, A, C, col_name, out)

    print("\nConcluído.")


if __name__ == "__main__":
    main()

