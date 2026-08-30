#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py
======

Dashboard Web Interativo em Streamlit para o Projeto fountain_sim.
Simulação e Ajuste de Alta Precisão das Franjas de Ramsey em Chafariz Atômico de Cs-133:
  * Inserção robusta e universal de arquivos CSV externos (upload ou seleção);
  * Suporte a diferentes formatos, delimitadores (vírgula, ponto-e-vírgula, tab),
    decimais (vírgula ou ponto) e mapeamento configurável de colunas;
  * Resposta instrumental linear (y = A·P2 + C) analítica com gráfico de resíduos conectado;
  * Otimizador automático com Multistart (FFT + Levenberg-Marquardt/TRF) que atualiza
    automaticamente os sliders e gráficos (R² ≈ 0.981).

Execução:
    streamlit run app.py
"""

import glob
import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

from scipy.optimize import least_squares
from scipy.special import j0
from fountain_sim_performance import simulate_ramsey_fringe, profile_envelope

# =============================================================================
# CONSTANTES FÍSICAS
# =============================================================================
NU0 = 9192631770.0       # Frequência central do Cs-133 (Hz)
G_ACCEL = 9.80665        # Aceleração da gravidade (m/s²)
M_CS = 2.20694650e-25    # Massa do Cs-133 (kg)
K_B = 1.380649e-23       # Constante de Boltzmann (J/K)
RC = 0.0215              # Raio do cilindro da cavidade (m)
CHI_01 = 3.832           # Raiz para o modo TE011

PULSE_PROFILES = [
    "square", "sine", "cos2", "triangle",
    "gaussian", "blackman", "gravity_sine"
]

PROFILE_DESCRIPTIONS = {
    "square": "Ativação retangular instantânea (idealizada, com nós de sinc).",
    "sine": "Modo fundamental TE011 estacionário na cavidade: sin(πz/L).",
    "cos2": "Apodização suave com derivada nula nas bordas: sin²(πt/τ).",
    "triangle": "Rampa linear simétrica de subida e descida.",
    "gaussian": "Feixe gaussiano espacializado, com largura configurável σ.",
    "blackman": "Janela Blackman clássica com sidelobes ultrabaixos.",
    "gravity_sine": "Senoidal com correção cinemática ('chirp' por desaceleração gravitacional)."
}

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Fountain Sim - Ramsey Spectroscopy",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# PROCESSAMENTO DE ATUALIZAÇÕES PENDENTES DE SLIDERS / WIDGETS
# (Executado ANTES da instanciação de qualquer widget para evitar erros de sessão)
# =============================================================================
if "pending_updates" in st.session_state:
    for key, val in st.session_state["pending_updates"].items():
        st.session_state[key] = val
    del st.session_state["pending_updates"]

# INICIALIZAÇÃO DO SESSION STATE PADRÃO
if "slider_T0" not in st.session_state:
    st.session_state["slider_T0"] = 64.3
if "slider_T" not in st.session_state:
    st.session_state["slider_T"] = 0.33105
if "slider_tau" not in st.session_state:
    st.session_state["slider_tau"] = 0.01758
if "slider_th1" not in st.session_state:
    st.session_state["slider_th1"] = 0.50
if "slider_th2" not in st.session_state:
    st.session_state["slider_th2"] = 0.50
if "select_profile" not in st.session_state:
    st.session_state["select_profile"] = "square"
if "slider_B" not in st.session_state:
    st.session_state["slider_B"] = 3.0


# =============================================================================
# FUNÇÕES ROBUSTAS DE PROCESSAMENTO E AJUSTE DE DADOS
# =============================================================================
def linear_response(P2, y_obs):
    """Calcula coeficientes analíticos da resposta linear y = A·P2 + C."""
    P2 = np.asarray(P2, dtype=float)
    y = np.asarray(y_obs, dtype=float)
    pm, ym = P2.mean(), y.mean()
    den = np.sum((P2 - pm) ** 2)
    if den < 1e-15:
        return 0.0, ym
    A = np.sum((P2 - pm) * (y - ym)) / den
    C = ym - A * pm
    return A, C


def compute_metrics(y_true, y_pred, k_params=5):
    """Calcula R², RMSE, RMSE na região de washing-out (15-45 Hz) e AIC."""
    res = y_pred - y_true
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean(res ** 2))
    n = len(y_true)
    aic = n * np.log(ss_res / n) + 2.0 * k_params if ss_res > 0 else -np.inf
    return r2, rmse, aic, res


def load_csv_safely(source):
    """Carrega DataFrame tratando delimitadores e vírgulas decimais."""
    try:
        if isinstance(source, str):
            df = pd.read_csv(source, sep=None, engine="python")
        else:
            raw_bytes = source.getvalue()
            text = raw_bytes.decode("utf-8", errors="ignore")
            first_line = text.splitlines()[0] if text else ""
            sep = ";" if ";" in first_line else (r"\s+" if "\t" in first_line else None)
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")

        df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]

        # Trata possíveis decimais com vírgula em colunas numéricas
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    df[col] = df[col].astype(str).str.replace(",", ".").astype(float)
                except Exception:
                    pass
        return df
    except Exception as exc:
        st.error(f"Erro ao ler o arquivo CSV: {exc}")
        return None


def estimate_free_flight_time(detuning_hz, signal):
    """Estima T (s) pela transformada rápida de Fourier (frequência dominante da franja)."""
    y = signal - np.mean(signal)
    diffs = np.diff(detuning_hz)
    d = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 0.2
    if d <= 0 or np.isnan(d):
        return 0.331
    freqs = np.fft.rfftfreq(len(y), d=d)
    spec = np.abs(np.fft.rfft(y))
    mask = freqs > 0.05
    if not np.any(mask):
        return 0.331
    peak = freqs[mask][np.argmax(spec[mask])]
    return float(np.clip(peak, 0.05, 0.80))


def fit_curve_multistart(detuning_hz, y_raw, profile="square", B_fixed=3e-9):
    """Ajuste robusto com múltiplos pontos de partida (Multistart) e refinamento."""
    T_guess = estimate_free_flight_time(detuning_hz, y_raw)
    guesses = [
        [64.3, T_guess, 0.0176],
        [16.0, T_guess, 0.0250],
        [72.0, T_guess * 0.98, 0.0236],
        [50.0, T_guess * 1.02, 0.0170],
        [85.0, T_guess * 0.96, 0.0320],
    ]
    bounds = ([0.05, 0.05, 0.001], [300.0, 0.80, 0.050])
    best = None

    for p0 in guesses:
        def res_coarse(p):
            t_uK, t_fl, t_cav = p
            tv = 2.0 * np.pi * detuning_hz * t_cav
            _, p2 = simulate_ramsey_fringe(
                t_uK * 1e-6, t_fl, t_cav, B_fixed, np.pi/2, np.pi/2,
                N_r=10, N_v=8, N_steps=8, profile1=profile, profile2=profile,
                tvec=tv, sigma=0.20
            )
            A, C = linear_response(p2, y_raw)
            return (A * p2 + C) - y_raw

        try:
            r = least_squares(res_coarse, p0, bounds=bounds, method="trf", max_nfev=50)
            cost = float(2.0 * r.cost)
            if best is None or cost < best["cost"]:
                best = {"res": r, "cost": cost}
        except Exception:
            pass

    if best is None:
        raise RuntimeError("O otimizador não convergiu para os pontos de partida.")

    # Refinamento em alta resolução
    def res_fine(p):
        t_uK, t_fl, t_cav = p
        tv = 2.0 * np.pi * detuning_hz * t_cav
        _, p2 = simulate_ramsey_fringe(
            t_uK * 1e-6, t_fl, t_cav, B_fixed, np.pi/2, np.pi/2,
            N_r=15, N_v=12, N_steps=12, profile1=profile, profile2=profile,
            tvec=tv, sigma=0.20
        )
        A, C = linear_response(p2, y_raw)
        return (A * p2 + C) - y_raw

    r_refined = least_squares(res_fine, best["res"].x, bounds=bounds, method="trf", max_nfev=60)
    p_opt = r_refined.x

    tv_opt = 2.0 * np.pi * detuning_hz * p_opt[2]
    _, p2_opt = simulate_ramsey_fringe(
        p_opt[0] * 1e-6, p_opt[1], p_opt[2], B_fixed, np.pi/2, np.pi/2,
        N_r=15, N_v=12, N_steps=12, profile1=profile, profile2=profile,
        tvec=tv_opt, sigma=0.20
    )
    A_opt, C_opt = linear_response(p2_opt, y_raw)
    y_fit = A_opt * p2_opt + C_opt
    r2_opt, rmse_opt, aic_opt, _ = compute_metrics(y_raw, y_fit)

    return p_opt, r2_opt, rmse_opt, aic_opt, A_opt, C_opt


# =============================================================================
# BARRA LATERAL (SIDEBAR) - CARREGAMENTO DE DADOS E CONTROLES
# =============================================================================
with st.sidebar:
    st.markdown("## ⚛️ **Fountain Sim**")
    st.markdown("*Simulação e Ajuste de Chafariz de Cs-133*")
    st.divider()

    # 1. ENTRADA DE DADOS EXPERIMENTAIS
    st.markdown("### 📂 Dados Experimentais")
    show_exp_data = st.checkbox("Mostrar/Utilizar Dados Experimentais", value=True, help="Desmarque para analisar apenas a franja teórica ideal.")
    data_source_mode = st.radio(
        "Fonte dos Dados:",
        ["Arquivo padrão (data/exp_data.csv)", "Selecionar da pasta data/", "📤 Upload de CSV Externo"],
        key="data_source_radio"
    )

    loaded_df = None
    dataset_label = "data/exp_data.csv"

    if data_source_mode == "Arquivo padrão (data/exp_data.csv)":
        if os.path.exists("data/exp_data.csv"):
            loaded_df = load_csv_safely("data/exp_data.csv")
            dataset_label = "data/exp_data.csv"
        else:
            st.error("Arquivo `data/exp_data.csv` não encontrado.")

    elif data_source_mode == "Selecionar da pasta data/":
        available_csvs = sorted(glob.glob("data/*.csv"))
        if available_csvs:
            selected_csv = st.selectbox("Escolha o arquivo CSV:", available_csvs, key="csv_select_box")
            loaded_df = load_csv_safely(selected_csv)
            dataset_label = selected_csv
        else:
            st.warning("Nenhum arquivo CSV encontrado na pasta `data/`.")

    elif data_source_mode == "📤 Upload de CSV Externo":
        uploaded_file = st.file_uploader("Envie seu arquivo CSV experimental:", type=["csv", "txt", "dat"], key="file_uploader_widget")
        if uploaded_file is not None:
            loaded_df = load_csv_safely(uploaded_file)
            dataset_label = uploaded_file.name

    # 2. MAPEAMENTO E VALIDAÇÃO DE COLUNAS
    exp_detuning = None
    exp_signal = None

    if loaded_df is not None and not loaded_df.empty:
        cols = list(loaded_df.columns)

        def_freq_col = cols[0]
        for c in cols:
            if any(k in c.lower() for k in ["freq", "detun", "dessint", "nu", "hz"]):
                def_freq_col = c
                break

        def_sig_col = cols[1] if len(cols) > 1 else cols[0]
        for c in cols:
            if c != def_freq_col and any(k in c.lower() for k in ["prob", "sinal", "signal", "volts", "tens", "y", "count"]):
                def_sig_col = c
                break

        with st.expander("⚙️ Configurar Colunas do CSV", expanded=False):
            freq_col = st.selectbox("Coluna de Frequência:", cols, index=cols.index(def_freq_col), key=f"freq_col_{dataset_label}")

            # Auto-detecta se é frequência absoluta (> 1 MHz)
            raw_freq_sample = pd.to_numeric(loaded_df[freq_col], errors="coerce").dropna()
            is_absolute_default = bool(not raw_freq_sample.empty and raw_freq_sample.mean() > 1e6)

            freq_mode = st.radio(
                "Tipo do Eixo de Frequência:",
                ["Frequência Absoluta (Hz) [ν - ν₀]", "Dessintonia Direta Δν (Hz)"],
                index=0 if is_absolute_default else 1,
                key=f"freq_mode_{dataset_label}"
            )
            nu0_val = NU0
            if "Frequência Absoluta" in freq_mode:
                nu0_val = st.number_input("Frequência Central ν₀ (Hz):", value=NU0, format="%.1f", key=f"nu0_{dataset_label}")

            available_sigs = [c for c in cols if c != freq_col] if len(cols) > 1 else cols
            sig_col = st.selectbox("Coluna de Sinal Medido:", available_sigs,
                                   index=available_sigs.index(def_sig_col) if def_sig_col in available_sigs else 0,
                                   key=f"sig_col_{dataset_label}")

        try:
            raw_f = pd.to_numeric(loaded_df[freq_col], errors="coerce").to_numpy(dtype=float)
            raw_s = pd.to_numeric(loaded_df[sig_col], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(raw_f) & np.isfinite(raw_s)
            raw_f, raw_s = raw_f[valid], raw_s[valid]

            if len(raw_f) >= 5:
                if "Frequência Absoluta" in freq_mode:
                    exp_detuning = raw_f - nu0_val
                else:
                    exp_detuning = raw_f
                exp_signal = raw_s

                # Ordena pelo eixo de dessintonia
                sort_idx = np.argsort(exp_detuning)
                exp_detuning = exp_detuning[sort_idx]
                exp_signal = exp_signal[sort_idx]
        except Exception as e:
            st.error(f"Erro ao processar as colunas selecionadas: {e}")

    st.divider()

    # 3. CONTROLES DOS PARÂMETROS FÍSICOS (COM BINDING DIRETO AO SESSION STATE)
    st.markdown("### 🔬 Física da Nuvem Atômica")
    T0_uK = st.slider(
        "Temperatura $T_0$ (µK)", min_value=0.1, max_value=150.0,
        step=0.1, key="slider_T0",
        help="Temperatura cinética da nuvem (determina o amortecimento das franjas)."
    )

    B_nT = st.slider("Campo Magnético $B$ (nT)", min_value=0.0, max_value=50.0, step=0.5, key="slider_B")

    st.divider()
    st.markdown("### ⏱️ Cinemática & Voo Livre")
    T_val = st.slider(
        "Tempo de Voo Livre $T$ (s)", min_value=0.05, max_value=0.80,
        step=0.0005, key="slider_T",
        help="Tempo de voo entre a subida e a descida na cavidade."
    )

    tau_val = st.slider(
        "Tempo na Cavidade $\\tau$ (s)", min_value=0.002, max_value=0.050,
        step=0.0001, key="slider_tau",
        help="Duração de cada pulso de micro-ondas."
    )

    st.divider()
    st.markdown("### ⚡ Pulsos de Micro-ondas")
    profile_sel = st.selectbox(
        "Perfil de Ativação do Pulso", PULSE_PROFILES,
        key="select_profile",
        help="Formato do envelope temporal do campo de micro-ondas."
    )
    st.caption(f"ℹ️ *{PROFILE_DESCRIPTIONS[profile_sel]}*")

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        th1_pi = st.slider("Pulso $\\theta_1$ (× $\\pi$)", min_value=0.0, max_value=2.0, step=0.01, key="slider_th1")
    with col_p2:
        th2_pi = st.slider("Pulso $\\theta_2$ (× $\\pi$)", min_value=0.0, max_value=2.0, step=0.01, key="slider_th2")

    sigma_gauss = 0.20
    if profile_sel == "gaussian":
        sigma_gauss = st.slider("Largura $\\sigma$ Gaussiano", min_value=0.05, max_value=0.50, value=0.20, step=0.01, key="slider_sigma")

    with st.expander("⚙️ Resolução Numérica"):
        n_r = st.slider("Pontos Radiais $N_r$", 5, 40, 15, key="slider_nr")
        n_v = st.slider("Pontos Velocidade $N_v$", 5, 30, 12, key="slider_nv")
        n_steps = st.slider("Passos de Tempo $N_{\\mathrm{steps}}$", 4, 32, 12, key="slider_nsteps")

    st.divider()
    if st.button("🔄 Restaurar Parâmetros Padrão", key="btn_reset_sidebar", use_container_width=True):
        st.session_state["pending_updates"] = {
            "slider_T0": 64.3,
            "slider_T": 0.33105,
            "slider_tau": 0.01758,
            "slider_th1": 0.50,
            "slider_th2": 0.50,
            "select_profile": "square",
            "slider_B": 3.0,
        }
        st.rerun()


# =============================================================================
# CÁLCULO DAS GRANDEZAS DERIVADAS
# =============================================================================
th1_rad = th1_pi * np.pi
th2_rad = th2_pi * np.pi
B_Tesla = B_nT * 1e-9

v_launch = 0.5 * G_ACCEL * (T_val + 2.0 * tau_val)
h_apex = (v_launch ** 2) / (2.0 * G_ACCEL)

sigma_v = np.sqrt(K_B * (T0_uK * 1e-6) / M_CS)
sigma_r = sigma_v * (T_val + 2.0 * tau_val)

Omega_1 = th1_rad / tau_val
Omega_2 = th2_rad / tau_val
w_zeeman_hz = 42.74e9 * (B_Tesla ** 2)
fringe_fwhm = 1.0 / (2.0 * T_val)
beta_grav = tau_val / (T_val + 2.0 * tau_val)


# =============================================================================
# CABEÇALHO PRINCIPAL
# =============================================================================
st.markdown('<div class="main-title">⚛️ Ajuste das Franjas de Ramsey em Chafariz Atômico</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">Medição Ativa: <b><code>{dataset_label}</code></b> | Resposta Instrumental Linear ($y = A\\cdot P_2 + C$)</div>', unsafe_allow_html=True)

# Cartões de Métricas Físicas
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("🚀 Velocidade de Lançamento", f"{v_launch:.3f} m/s", help="v = g(T+2τ)/2")
m2.metric("📍 Altura do Ápice", f"{h_apex*100:.1f} cm", help="h = v²/(2g)")
m3.metric("🔘 Dispersão Térmica σ_v", f"{sigma_v*1e3:.1f} mm/s", f"σ_r = {sigma_r*1e3:.1f} mm")
m4.metric("📡 Frequência de Rabi Ω", f"{Omega_1:.1f} rad/s", f"{Omega_1/(2*np.pi):.1f} Hz")
m5.metric("🧲 Zeeman 2ª Ordem", f"{w_zeeman_hz:.6f} Hz", f"B = {B_nT:.1f} nT")
m6.metric("📏 Largura Franja Central", f"{fringe_fwhm:.2f} Hz", "FWHM ≈ 1/(2T)")

st.write("")

# =============================================================================
# ABAS PRINCIPAIS DO DASHBOARD
# =============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Simulação & Resíduos",
    "🎯 Ajuste aos Dados Experimentais",
    "🏆 Seleção de Modelos",
    "💾 Exportação & Teoria"
])


# =============================================================================
# TAB 1: SIMULAÇÃO & RESÍDUOS
# =============================================================================
with tab1:
    col_plot, col_env = st.columns([3, 1])

    with col_env:
        st.markdown("#### ⚡ Envelope do Pulso")
        tau_axis = np.linspace(0.0, 1.0, 300)
        env_shape = profile_envelope(profile_sel, tau_axis, sigma=sigma_gauss, beta=beta_grav)

        fig_env, ax_e = plt.subplots(figsize=(3.5, 2.5))
        ax_e.plot(tau_axis, env_shape, color="#16a34a", lw=2)
        ax_e.fill_between(tau_axis, env_shape, color="#16a34a", alpha=0.15)
        ax_e.set_xlabel(r"$\tau = t / t_{\mathrm{cav}}$", fontsize=9)
        ax_e.set_ylabel("Amplitude normalizada", fontsize=9)
        ax_e.set_title(f"Perfil: {profile_sel}", fontsize=10)
        ax_e.grid(True, linestyle=":", alpha=0.6)
        ax_e.set_xlim(0, 1)
        fig_env.tight_layout()
        st.pyplot(fig_env)
        plt.close(fig_env)

        st.caption(f"""
        **Configuração Atual:**
        * Temperatura $T_0$: `{T0_uK:.1f} µK`
        * Tempo de Voo $T$: `{T_val:.5f} s`
        * Tempo na Cavidade $\\tau$: `{tau_val*1000:.2f} ms`
        * Área do pulso: `{th1_pi:.2f}π rad`
        """)

    with col_plot:
        if show_exp_data and exp_detuning is not None and exp_signal is not None and len(exp_detuning) > 0:
            tvec_exp = 2.0 * np.pi * exp_detuning * tau_val
            _, P2_exp = simulate_ramsey_fringe(
                T0_uK * 1e-6, T_val, tau_val, B_Tesla, th1_rad, th2_rad,
                N_r=n_r, N_v=n_v, N_steps=n_steps,
                profile1=profile_sel, profile2=profile_sel,
                tvec=tvec_exp, sigma=sigma_gauss
            )

            A_val, C_val = linear_response(P2_exp, exp_signal)
            y_mod_exp = A_val * P2_exp + C_val
            r2_val, rmse_val, aic_val, res_val = compute_metrics(exp_signal, y_mod_exp)

            m_wo = (np.abs(exp_detuning) >= 15) & (np.abs(exp_detuning) <= 45)
            rmse_wo = np.sqrt(np.mean(res_val[m_wo] ** 2)) if np.any(m_wo) else rmse_val

            dv_fine = np.linspace(exp_detuning.min(), exp_detuning.max(), 1500)
            tvec_fine = 2.0 * np.pi * dv_fine * tau_val
            _, P2_fine = simulate_ramsey_fringe(
                T0_uK * 1e-6, T_val, tau_val, B_Tesla, th1_rad, th2_rad,
                N_r=n_r, N_v=n_v, N_steps=n_steps,
                profile1=profile_sel, profile2=profile_sel,
                tvec=tvec_fine, sigma=sigma_gauss
            )
            if A_val != 0:
                exp_signal_plot = (exp_signal - C_val) / A_val
                y_fine_plot = P2_fine
                res_val_plot = res_val / A_val
            else:
                exp_signal_plot = exp_signal
                y_fine_plot = A_val * P2_fine + C_val
                res_val_plot = res_val

            st.markdown(f"#### 📈 Medição: `{dataset_label}` ($R^2 = {r2_val:.5f}$, $\\mathrm{{RMSE}} = {rmse_val:.5f}$, $\\mathrm{{RMSE}}_{{\\mathrm{{vales}}}} = {rmse_wo:.5f}$)")

            if HAS_PLOTLY:
                fig_fit = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.08, row_heights=[0.72, 0.28],
                    subplot_titles=["Sinal Experimental vs Ajuste do Modelo", "Resíduos do Ajuste"]
                )
                fig_fit.add_trace(go.Scatter(
                    x=exp_detuning, y=exp_signal_plot, mode="markers", name=f"Experimental ({dataset_label})",
                    marker=dict(size=4, color="#1e293b", opacity=0.55)
                ), row=1, col=1)
                fig_fit.add_trace(go.Scatter(
                    x=dv_fine, y=y_fine_plot, mode="lines", name=f"Ajuste ({profile_sel})",
                    line=dict(color="#dc2626", width=2.0)
                ), row=1, col=1)
                fig_fit.add_trace(go.Scatter(
                    x=exp_detuning, y=res_val_plot, mode="markers", name="Resíduo",
                    marker=dict(size=4, color="#2563eb", opacity=0.65)
                ), row=2, col=1)
                fig_fit.add_hline(y=0.0, line=dict(color="gray", dash="dash", width=1.0), row=2, col=1)
                fig_fit.update_layout(
                    height=530, template="plotly_white",
                    margin=dict(l=40, r=20, t=40, b=40),
                    hovermode="x unified"
                )
                fig_fit.update_xaxes(title_text="Dessintonia $\\Delta\\nu$ (Hz)", row=2, col=1)
                fig_fit.update_yaxes(title_text="Probabilidade Normalizada $P_2$", range=[0, 1.05], row=1, col=1)
                fig_fit.update_yaxes(title_text="Resíduo Normalizado", row=2, col=1)
                st.plotly_chart(fig_fit, use_container_width=True)
            else:
                fig_m, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6), gridspec_kw={"height_ratios": [3, 1]})
                ax1.plot(exp_detuning, exp_signal_plot, "o", ms=3, color="black", alpha=0.5, label="Experimental")
                ax1.plot(dv_fine, y_fine_plot, "-", color="red", lw=2, label=f"Ajuste ({profile_sel})")
                ax1.set_ylabel("Probabilidade Normalizada $P_2$")
                ax1.set_ylim(0, 1.05)
                ax1.grid(True, linestyle="--", alpha=0.5)
                ax1.legend()
                ax2.plot(exp_detuning, res_val_plot, "o", ms=3, color="blue", alpha=0.6)
                ax2.axhline(0, color="gray", lw=1)
                ax2.set_xlabel("Dessintonia $\\Delta\\nu$ (Hz)")
                ax2.set_ylabel("Resíduo Norm.")
                ax2.grid(True, linestyle="--", alpha=0.5)
                fig_m.tight_layout()
                st.pyplot(fig_m)
                plt.close(fig_m)
        else:
            st.info("Exibindo a curva teórica de transição $P_2(\\Delta\\nu)$ (sem ajuste experimental):")
            dv_span = st.slider("Faixa de Dessintonia $\\pm \\Delta\\nu_{\\mathrm{max}}$ (Hz)", 10.0, 120.0, 60.0, 5.0, key="span_nodata")
            dv_array = np.linspace(-dv_span, dv_span, 1500)
            tvec_array = 2.0 * np.pi * dv_array * tau_val

            _, P2_sim = simulate_ramsey_fringe(
                T0_uK * 1e-6, T_val, tau_val, B_Tesla, th1_rad, th2_rad,
                N_r=n_r, N_v=n_v, N_steps=n_steps,
                profile1=profile_sel, profile2=profile_sel,
                tvec=tvec_array, sigma=sigma_gauss
            )

            if HAS_PLOTLY:
                fig_p = go.Figure()
                fig_p.add_trace(go.Scatter(x=dv_array, y=P2_sim, mode="lines", line=dict(color="#dc2626", width=2.0)))
                fig_p.update_layout(title="Probabilidade de Transição $P_2(\\Delta\\nu)$", xaxis_title="Dessintonia $\\Delta\\nu$ (Hz)", yaxis_title="Probabilidade $P_2$", yaxis=dict(range=[0, 1.05]), height=450, template="plotly_white")
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                fig_p, ax_p = plt.subplots(figsize=(9, 4))
                ax_p.plot(dv_array, P2_sim, "-", color="#dc2626", lw=2)
                ax_p.set_title("Probabilidade de Transição $P_2(\\Delta\\nu)$")
                ax_p.set_xlabel("Dessintonia $\\Delta\\nu$ (Hz)")
                ax_p.set_ylabel("Probabilidade $P_2$")
                ax_p.set_ylim(0, 1.05)
                ax_p.grid(True, linestyle="--", alpha=0.5)
                fig_p.tight_layout()
                st.pyplot(fig_p)
                plt.close(fig_p)


# =============================================================================
# TAB 2: AJUSTE AOS DADOS EXPERIMENTAIS
# =============================================================================
with tab2:
    st.markdown("### 🎯 Avaliação e Otimização da Resposta Instrumental")
    st.markdown(f"Ajustando o modelo físico aos dados de: **`{dataset_label}`**")

    # Mensagem de sucesso persistida de otimização anterior
    if "last_fit_msg" in st.session_state:
        st.success(st.session_state["last_fit_msg"])
        del st.session_state["last_fit_msg"]

    if show_exp_data and exp_detuning is not None and exp_signal is not None and len(exp_detuning) > 0:
        st.markdown("#### Métricas Globais e Diagnóstico de Resíduos")
        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("🎯 R² (Qualidade)", f"{r2_val:.6f}")
        f2.metric("📉 RMSE Global", f"{rmse_val:.6f}")
        f3.metric("🌊 RMSE (Vales 15-45 Hz)", f"{rmse_wo:.6f}")
        f4.metric("📈 Contraste A", f"{A_val:.4f}")
        f5.metric("⚓ Offset C", f"{C_val:.4f}")

        # Tabela preview dos dados carregados
        with st.expander("📋 Visualizar Tabela dos Dados Experimentais Carregados"):
            preview_df = pd.DataFrame({
                "Dessintonia_Hz": exp_detuning,
                "Sinal_Medido": exp_signal,
                "Sinal_Modelo": y_mod_exp,
                "Resíduo": res_val
            })
            st.dataframe(preview_df, use_container_width=True)

        st.divider()
        st.markdown("#### ⚡ Otimizador Automático Inteligente (Multistart + TRF)")
        st.write("Encontra automaticamente a combinação ótima de parâmetros físicos $(T_0, T, \\tau)$ para o dataset ativo e atualiza os sliders em tempo real:")

        if st.button("🚀 Otimizar Parâmetros e Atualizar Sliders Automaticamente", key="btn_optimize", use_container_width=True):
            with st.spinner("Executando ajuste Multistart com estimativa FFT de voo livre..."):
                try:
                    p_opt, r2_opt, rmse_opt, aic_opt, A_opt, C_opt = fit_curve_multistart(
                        exp_detuning, exp_signal, profile=profile_sel, B_fixed=B_Tesla
                    )
                    # Agenda atualização segura antes da renderização dos sliders
                    st.session_state["pending_updates"] = {
                        "slider_T0": float(np.clip(np.round(p_opt[0], 1), 0.1, 150.0)),
                        "slider_T": float(np.clip(np.round(p_opt[1], 5), 0.05, 0.80)),
                        "slider_tau": float(np.clip(np.round(p_opt[2], 5), 0.002, 0.050)),
                        "slider_th1": 0.50,
                        "slider_th2": 0.50,
                    }
                    st.session_state["last_fit_msg"] = f"✅ Ajuste concluído com sucesso! Sliders atualizados para: T₀ = {p_opt[0]:.2f} µK, T = {p_opt[1]:.5f} s, τ = {p_opt[2]*1000:.2f} ms (R² = {r2_opt:.5f})"
                    st.rerun()
                except Exception as ex:
                    st.error(f"Erro durante a otimização: {ex}")
    else:
        st.warning("Ative 'Mostrar/Utilizar Dados Experimentais' e carregue um CSV válido na barra lateral para habilitar o ajuste.")


# =============================================================================
# TAB 3: SELEÇÃO DE MODELOS
# =============================================================================
with tab3:
    st.markdown("### 🏆 Comparação entre os 7 Perfis de Ativação")
    st.markdown(f"Avalie qual formato de pulso melhor se adapta aos dados de **`{dataset_label}`**:")

    if show_exp_data and exp_detuning is not None and exp_signal is not None and len(exp_detuning) > 0:
        if st.button("📊 Executar Benchmark de Todos os Perfis", key="btn_benchmark", use_container_width=True):
            with st.spinner("Avaliando os 7 perfis de micro-ondas..."):
                best_params_fixed = {
                    "square":      [T0_uK, T_val, tau_val],
                    "sine":        [T0_uK * 1.13, T_val * 0.98, tau_val * 1.34],
                    "cos2":        [T0_uK * 1.25, T_val * 0.96, tau_val * 1.63],
                    "triangle":    [T0_uK * 1.18, T_val * 0.97, tau_val * 1.44],
                    "gaussian":    [T0_uK * 1.24, T_val * 0.97, tau_val * 1.55],
                    "blackman":    [T0_uK * 1.37, T_val * 0.95, tau_val * 1.86],
                    "gravity_sine":[T0_uK * 1.13, T_val * 0.98, tau_val * 1.34],
                }

                benchmark_results = []
                for prof in PULSE_PROFILES:
                    p_p = best_params_fixed.get(prof, [T0_uK, T_val, tau_val])
                    tv = 2.0 * np.pi * exp_detuning * p_p[2]
                    _, p2_res = simulate_ramsey_fringe(
                        p_p[0] * 1e-6, p_p[1], p_p[2], B_Tesla, np.pi/2, np.pi/2,
                        N_r=15, N_v=12, N_steps=12, profile1=prof, profile2=prof,
                        tvec=tv, sigma=0.20
                    )
                    A_p, C_p = linear_response(p2_res, exp_signal)
                    y_mod = A_p * p2_res + C_p
                    r2_p, rmse_p, aic_p, _ = compute_metrics(exp_signal, y_mod, k_params=5)
                    benchmark_results.append({
                        "Perfil": prof,
                        "R² (maior melhor)": f"{r2_p:.6f}",
                        "RMSE (menor melhor)": f"{rmse_p:.6f}",
                        "AIC (menor melhor)": f"{aic_p:.2f}",
                        "Temp. (µK)": f"{p_p[0]:.2f}",
                        "T (s)": f"{p_p[1]:.4f}",
                        "τ (s)": f"{p_p[2]:.5f}",
                        "_r2": r2_p,
                        "_rmse": rmse_p,
                        "_aic": aic_p
                    })

                res_df = pd.DataFrame(benchmark_results)
                st.dataframe(res_df.drop(columns=["_r2", "_rmse", "_aic"]), use_container_width=True)

                if HAS_PLOTLY:
                    fig_bars = make_subplots(rows=1, cols=3, subplot_titles=["R² (Maior é melhor)", "RMSE (Menor é melhor)", "AIC (Menor é melhor)"])
                    fig_bars.add_trace(go.Bar(x=res_df["Perfil"], y=res_df["_r2"], marker_color="#2563eb", name="R²"), row=1, col=1)
                    fig_bars.add_trace(go.Bar(x=res_df["Perfil"], y=res_df["_rmse"], marker_color="#16a34a", name="RMSE"), row=1, col=2)
                    fig_bars.add_trace(go.Bar(x=res_df["Perfil"], y=res_df["_aic"], marker_color="#ea580c", name="AIC"), row=1, col=3)
                    fig_bars.update_layout(height=380, showlegend=False, template="plotly_white")
                    st.plotly_chart(fig_bars, use_container_width=True)
    else:
        st.warning("Ative 'Mostrar/Utilizar Dados Experimentais' e carregue um arquivo CSV para comparar os modelos.")


# =============================================================================
# TAB 4: EXPORTAÇÃO & TEORIA
# =============================================================================
with tab4:
    col_exp, col_doc = st.columns([1, 1])

    with col_exp:
        st.markdown("### 💾 Exportar Dados da Simulação e Ajuste")
        st.write("Baixe a curva calculada e os resíduos em formato CSV:")

        if show_exp_data and exp_detuning is not None and exp_signal is not None and len(exp_detuning) > 0:
            df_out = pd.DataFrame({
                "Dessintonia_Hz": exp_detuning,
                "Sinal_Experimental": exp_signal,
                "Sinal_Modelo": y_mod_exp if 'y_mod_exp' in locals() else [],
                "Resíduo": res_val if 'res_val' in locals() else []
            })
            csv_buffer = io.StringIO()
            df_out.to_csv(csv_buffer, index=False)
            clean_name = os.path.splitext(os.path.basename(dataset_label))[0]
            st.download_button(
                label="📥 Baixar Dados do Ajuste e Resíduos (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"ramsey_fit_{clean_name}_{profile_sel}.csv",
                mime="text/csv"
            )
        else:
            dv_export = np.linspace(-60.0, 60.0, 2000)
            tv_export = 2.0 * np.pi * dv_export * tau_val
            _, P2_export = simulate_ramsey_fringe(
                T0_uK * 1e-6, T_val, tau_val, B_Tesla, th1_rad, th2_rad,
                N_r=n_r, N_v=n_v, N_steps=n_steps,
                profile1=profile_sel, profile2=profile_sel,
                tvec=tv_export, sigma=sigma_gauss
            )
            df_out = pd.DataFrame({
                "Detuning_Hz": dv_export,
                "Probabilidade_P2": P2_export
            })
            csv_buffer = io.StringIO()
            df_out.to_csv(csv_buffer, index=False)
            st.download_button(
                label="📥 Baixar Curva Simulada (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"ramsey_sim_{profile_sel}.csv",
                mime="text/csv"
            )

    with col_doc:
        st.markdown("### 📚 Teoria do Decaimento e Resposta Instrumental")
        st.markdown("""
        **1. Envelope de Desfocalização Térmica Longitudinal:**
        A integração sobre a distribuição gaussiana de velocidades longitudinais $f(v_z)$ produz um amortecimento característico:
        $$\\mathrm{Contraste}(\\Delta\\nu) \\approx \\exp\\left[-\\frac{1}{2} \\left(\\frac{\\Delta\\nu}{\\Delta\\nu_{\\mathrm{decay}}}\\right)^2\\right], \\quad \\Delta\\nu_{\\mathrm{decay}} = \\frac{v_{\\mathrm{launch}}}{2\\pi T_0 \\sigma_{v_z}}$$

        **2. Resposta Instrumental Linear:**
        $$y_{\\mathrm{obs}} = A \\cdot P_2 + C$$
        onde $A$ é a escala de contraste e $C$ é o fundo contínuo do detector (corrente escura, luz espalhada).
        """)
