#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fountain_sim_performance.py
===========================

Versão OTIMIZADA da simulação das franjas de Ramsey com pulso NÃO-instantâneo
(perfil seno/Blackman do campo de micro-ondas na cavidade).

Mantém a MESMA física de ``fountain_sim_alt.py`` (rotação de Bloch XZ, perfil
espacial Bessel J0, dispersão térmica transversal e longitudinal), porém com
as seguintes otimizações numéricas:

  1. Avaliação direta nas dessintonias pedidas (``tvec``), em vez de varrer a
     faixa completa e interpolar;
  2. Quadratura de Gauss-Hermite para a integral de velocidade (peso gaussiano);
  3. Quadratura de Gauss-Legendre para a integral radial em [0, Rc];
  4. Integração do pulso com ``N_steps`` configurável (regra do ponto médio);
  5. Rotação de Bloch XZ vetorizada, sem alocação de memória desnecessária.

Isso reduz drasticamente o custo computacional preservando (ou melhorando) a
precisão da integração térmica, permitindo usar o pulso modelado dentro de um
laço de ajuste em tempo razoável.
"""

import numpy as np
from scipy.special import j0


def apply_bloch_rotation_XZ(state, Omega_x, Omega_z, t):
    """
    Rotação de Bloch otimizada assumindo campo micro-ondas apenas no eixo X
    e dessintonia apenas no eixo Z (componente Y nula). Vetorizada.
    """
    W_norm = np.sqrt(Omega_x**2 + Omega_z**2)
    W_norm = np.where(W_norm == 0, 1e-15, W_norm)

    kx = Omega_x / W_norm
    kz = Omega_z / W_norm

    x, y, z = state

    theta = W_norm * t
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    inv_cos = 1.0 - cos_t

    cross_x = -kz * y
    cross_y = kz * x - kx * z
    cross_z = kx * y
    dot_kv = kx * x + kz * z
    term3 = dot_kv * inv_cos

    new_x = x * cos_t + cross_x * sin_t + kx * term3
    new_y = y * cos_t + cross_y * sin_t
    new_z = z * cos_t + cross_z * sin_t + kz * term3

    return np.array([new_x, new_y, new_z])


def profile_envelope(profile_type, tau, sigma=0.2, beta=0.0):
    """
    Envelope temporal (não-instantâneo) do campo de micro-ondas no pulso.

    Perfis disponíveis (todos normalizados para ÁREA UNITÁRIA):
      * 'square'       : ativação instantânea (limite idealizado);
      * 'sine'         : modo fundamental TE011 (onda estacionária sin(πz/L));
      * 'cos2'         : apodização suave (derivada nula nas bordas);
      * 'triangle'     : rampa linear de subida/descida;
      * 'gaussian'     : campo localizado (feixe gaussiano), largura sigma;
      * 'gravity_sine' : senoidal com "chirp" gravitacional (átomo desacelera);
      * 'blackman'     : janela de sidelobes muito baixos.

    A normalização para média unitária garante que a área do pulso seja θ
    (Σ env·dtau = 1), preservando o significado físico de theta.
    """
    tau = np.asarray(tau, dtype=float)

    if profile_type == "square":
        env = np.ones_like(tau)
    elif profile_type == "sine":
        env = np.sin(np.pi * tau)
    elif profile_type == "cos2":
        env = np.sin(np.pi * tau) ** 2
    elif profile_type == "triangle":
        env = 1.0 - np.abs(2.0 * tau - 1.0)
    elif profile_type == "gaussian":
        env = np.exp(-((tau - 0.5) ** 2) / (2.0 * sigma ** 2))
    elif profile_type == "gravity_sine":
        env = np.sin(np.pi * (tau - beta * tau ** 2) / (1.0 - beta))
    elif profile_type == "blackman":
        env = 0.42 - 0.5 * np.cos(2.0 * np.pi * tau) + 0.08 * np.cos(4.0 * np.pi * tau)
    else:
        raise ValueError(f"Perfil de pulso desconhecido: {profile_type!r}")

    # Normaliza para área unitária (média = 1 => soma discreta * dtau = 1)
    return env / np.mean(env)


def simulate_ramsey_fringe(Temp_0, T, t, B=3e-9, theta1=np.pi / 2,
                           theta2=np.pi / 2, N_r=20, N_v=15, N_steps=16,
                           profile1="sine", profile2="sine", tvec=None,
                           sigma=0.2, h_mot=0.0, sigma_r0=0.0, R_aperture=None):
    """
    Simulação otimizada das franjas de Ramsey com pulso não-instantâneo e
    geometria experimental (MOT e aberturas de colimação).

    Parâmetros
    ----------
    Temp_0, T, t, B, theta1, theta2 : idênticos a ``fountain_sim_alt``.
    N_r  : número de pontos de Gauss-Legendre (integral radial, [0, R_max]).
    N_v  : número de pontos de Gauss-Hermite (integral de velocidade).
    N_steps : passos de integração do envelope do pulso (ponto médio).
    profile1, profile2 : perfil de ativação do micro-ondas (ex.: 'sine', 'blackman', 'square').
    tvec : (opcional) array de dessintonia normalizada (Delta*tau) onde avaliar.
    sigma : largura do feixe gaussiano (para profile='gaussian').
    h_mot : distância vertical da MOT/melaço óptico até a cavidade de micro-ondas (m).
    sigma_r0 : raio inicial da nuvem atômica no melaço (m).
    R_aperture : raio das aberturas de colimação da cavidade (m). Se None, usa Rc.

    Retorna
    -------
    tvec, P2
    """
    # Constantes físicas do experimento
    k_B = 1.380649e-23     # Constante de Boltzmann (J/K)
    m_cs = 2.20694650e-25  # Massa do Césio-133 (kg)
    chi_01 = 3.832         # Raiz para o modo TE011 da cavidade
    Rc = 0.0215            # Raio do cilindro da cavidade (m)
    g = 9.80665            # Gravidade (m/s^2)

    R_max = float(R_aperture) if (R_aperture is not None and R_aperture > 0) else Rc

    # Cinemática e frequências de Rabi
    v_cav = 0.5 * g * (T + 2.0 * t)
    if h_mot > 0.0:
        v_launch = np.sqrt(v_cav**2 + 2.0 * g * h_mot)
        t_subida = (v_launch - v_cav) / g
        amplif_v = v_launch / v_cav
    else:
        v_launch = v_cav
        t_subida = 0.0
        amplif_v = 1.0

    Omega_mw_1 = theta1 / t
    Omega_mw_2 = theta2 / t
    w_zeeman = 2.0 * np.pi * (42.74e9 * B**2)

    # ---- Eixo de dessintonia normalizada ---------------------------------
    if tvec is None:
        estimativa_franjas = (40.0 / t) / (2.0 * np.pi / T)
        points = int(max(500, 15 * estimativa_franjas))
        W0 = np.linspace(-20.0 / t, 20.0 / t, points)
        tvec = W0 * t
    else:
        tvec = np.asarray(tvec, dtype=float).ravel()

    delta = tvec / t + w_zeeman              # (n_det,)
    n_det = delta.size

    # ---- Dispersões térmicas ---------------------------------------------
    sigma_v = np.sqrt(k_B * Temp_0 / m_cs)
    t_total_flight = 2.0 * t_subida + T + 2.0 * t
    sigma_r = np.sqrt(sigma_r0**2 + (sigma_v * t_total_flight)**2)

    # ---- Quadratura de velocidade (Gauss-Hermite) ------------------------
    if sigma_v > 1e-9:
        x_v, w_v = np.polynomial.hermite.hermgauss(N_v)
        delta_v = np.sqrt(2.0) * sigma_v * x_v
        weights_v = w_v / np.sqrt(np.pi)     # normaliza o peso gaussiano
    else:
        delta_v = np.array([0.0])
        weights_v = np.array([1.0])

    # ---- Quadratura radial (Gauss-Legendre em [0, R_max]) ----------------
    x_r, w_r = np.polynomial.legendre.leggauss(N_r)
    r_grid = 0.5 * R_max * (1.0 + x_r)
    if sigma_r > 1e-9:
        weights_r = w_r * (0.5 * R_max) * r_grid * np.exp(-(r_grid**2) / (2.0 * sigma_r**2))
    else:
        weights_r = w_r * (0.5 * R_max) * r_grid
    s_r = weights_r.sum()
    if s_r <= 0.0:
        weights_r = np.full(N_r, 1.0 / N_r)
    else:
        weights_r = weights_r / s_r
    weights_v = weights_v / weights_v.sum()

    # ---- Malha 2D (velocidade x raio) ------------------------------------
    R_grid, V_grid = np.meshgrid(r_grid, delta_v)
    R_flat = R_grid.ravel()
    V_flat = V_grid.ravel()

    W_R, W_V = np.meshgrid(weights_r, weights_v)
    Weights_flat = (W_R * W_V).ravel()
    N_total = R_flat.size

    # ---- Perfil espacial (Bessel J0) e correção cinemática --------------
    Omega_local_1 = Omega_mw_1 * j0(chi_01 * R_flat / Rc)   # (N_total,)
    Omega_local_2 = Omega_mw_2 * j0(chi_01 * R_flat / Rc)
    V_eff = V_flat * amplif_v
    t_local = t * (1.0 - V_eff / v_launch)                 # (N_total,)
    T_local = T * (1.0 + V_eff / v_launch)

    # ---- Envelope do pulso (pré-computado) -------------------------------
    tau_mid = (np.arange(N_steps) + 0.5) / N_steps
    beta_gravity = t / (T + 2.0 * t)                       # "chirp" gravitacional
    env1 = profile_envelope(profile1, tau_mid, sigma=sigma, beta=beta_gravity)
    env2 = profile_envelope(profile2, tau_mid, sigma=sigma, beta=beta_gravity)
    dtau = 1.0 / N_steps

    # ---- Estado quântico inicial ----------------------------------------
    state = np.zeros((3, N_total, n_det))
    state[2] = 1.0

    delta_b = delta[np.newaxis, :]                          # (1, n_det)
    dt_local = (t_local * dtau)[:, np.newaxis]              # (N_total, 1)

    # 1º Pulso (subida pela cavidade)
    for e in env1:
        Omega_x = (Omega_local_1 * e)[:, np.newaxis]        # (N_total, 1)
        state = apply_bloch_rotation_XZ(state, Omega_x, delta_b, dt_local)

    # Voo livre (região sem micro-ondas)
    phi = delta_b * T_local[:, np.newaxis]
    x, y, z = state
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    state = np.array([x * cos_phi - y * sin_phi,
                      y * cos_phi + x * sin_phi,
                      z])

    # 2º Pulso (descida pela cavidade)
    for e in env2:
        Omega_x = (Omega_local_2 * e)[:, np.newaxis]
        state = apply_bloch_rotation_XZ(state, Omega_x, delta_b, dt_local)

    # Probabilidade de estado excitado, ponderada pelas distribuições térmicas
    P2_local = 0.5 * (1.0 - state[2])                       # (N_total, n_det)
    P2 = np.sum(P2_local * Weights_flat[:, np.newaxis], axis=0)

    return tvec, P2

