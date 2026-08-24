import numpy as np
from scipy.special import j0

# =============================================================================
# 1. FUNÇÕES DO MODELO FÍSICO
# =============================================================================

def apply_bloch_rotation(state, Omega_vec, t):
    """
    Aplica rotação no vetor de Bloch usando a Matriz de Rotação de Rodrigues 3x3.
    """
    # Frequência de Rabi generalizada (norma do vetor)
    W_norm = np.linalg.norm(Omega_vec, axis=0)
    W_norm = np.where(W_norm == 0, 1e-15, W_norm) # Evita divisão por zero
    
    # Componentes normalizadas do eixo de rotação
    nx, ny, nz = Omega_vec / W_norm
    
    # Ângulo de rotação (taxa de rotação * tempo de interação)
    theta = W_norm * t
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    inv_cos = 1 - cos_t
    
    # Elementos da Matriz de Rotação 3x3
    R11 = cos_t + nx**2 * inv_cos
    R12 = nx * ny * inv_cos - nz * sin_t
    R13 = nx * nz * inv_cos + ny * sin_t
    
    R21 = ny * nx * inv_cos + nz * sin_t
    R22 = cos_t + ny**2 * inv_cos
    R23 = ny * nz * inv_cos - nx * sin_t
    
    R31 = nz * nx * inv_cos - ny * sin_t
    R32 = nz * ny * inv_cos + nx * sin_t
    R33 = cos_t + nz**2 * inv_cos
    
    x, y, z = state
    
    # Novo estado evoluído após a multiplicação matricial
    new_x = R11 * x + R12 * y + R13 * z
    new_y = R21 * x + R22 * y + R23 * z
    new_z = R31 * x + R32 * y + R33 * z
    
    return np.array([new_x, new_y, new_z])

def simulate_ramsey_fringe(Temp_0, T, t, B=3e-9,theta1=np.pi/2,
                           theta2 = np.pi/2, N_r=50, N_v=30):
    """
    Simulação das franjas de Ramsey iterando sobre efeitos térmicos e espaciais.
    """
    # Constantes físicas do experimento
    k_B = 1.380649e-23     # Constante de Boltzmann (J/K)
    m_cs = 2.20694650e-25  # Massa do Césio-133 (kg)
    chi_01 = 3.832         # Raiz para o modo TE011 da cavidade
    Rc = 0.0215            # Raio da cavidade (m)
    g = 9.80665            # Gravidade (m/s^2)

    # Cinemática: velocidade necessária para atingir o tempo de voo T
    v_launch = 0.5 * g * (T + 2 * t)
    
    # Frequência de Rabi para um pulso determinado no centro das duas cavidades
    Omega_mw_1 = theta1 / t
    Omega_mw_2 = theta2 / t
    
    # Efeito Zeeman de 2ª ordem para o campo B fixado
    w_zeeman = 2 * np.pi * (42.74e9 * B**2)

    # Configuração da varredura do eixo X (dessintonia do micro-ondas)
    estimativa_franjas = (40 / t) / (2 * np.pi / T)
    points = int(max(500, 15 * estimativa_franjas)) 
    W_0_range = np.linspace(-20/t, 20/t, points)
    tvec = W_0_range * t
    P2 = np.zeros(points)

    # Dinâmica Térmica: Dispersão transversal (raio) devido à temperatura
    v_rms = np.sqrt(k_B * Temp_0 / m_cs)
    sigma_r = v_rms * (T + 2 * t)
    
    r_grid = np.linspace(1e-6, Rc, N_r)
    weights_r = r_grid * np.exp(- (r_grid**2) / (2 * sigma_r**2))
    weights_r = weights_r / np.sum(weights_r) if np.sum(weights_r) > 0 else np.ones_like(r_grid)/N_r

    # Dinâmica Térmica: Dispersão longitudinal (velocidade vertical)
    sigma_v = np.sqrt(k_B * Temp_0 / m_cs) 
    
    if sigma_v > 1e-6:
        delta_v_grid = np.linspace(-3*sigma_v, 3*sigma_v, N_v)
        weights_v = np.exp(- (delta_v_grid**2) / (2 * sigma_v**2))
        weights_v = weights_v / np.sum(weights_v)
    else: # Limite ideal (T = 0K)
        delta_v_grid = np.array([0.0])
        weights_v = np.array([1.0])

    # Criação da malha 2D englobando espaço (Raio) e momento (Velocidade)
    R_grid, V_grid = np.meshgrid(r_grid, delta_v_grid)
    R_flat = R_grid.flatten()
    V_flat = V_grid.flatten()
    
    W_R, W_V = np.meshgrid(weights_r, weights_v)
    Weights_flat = (W_R * W_V).flatten()

    # Perfil espacial do campo magnético de micro-ondas nas cavidades (Bessel J0)
    Omega_local_flat_1 = Omega_mw_1 * j0(chi_01 * R_flat / Rc)
    Omega_local_flat_2 = Omega_mw_2 * j0(chi_01 * R_flat / Rc)

    # Correção dos tempos de interação devido à dispersão térmica de velocidades
    t_local_flat = t * (1 - V_flat / v_launch)
    T_local_flat = T * (1 + V_flat / v_launch)

    # Estado quântico inicial: todos os átomos no estado fundamental (Pólo Sul, z = 1.0)
    N_total = len(R_flat)
    state_0_base = np.zeros((3, N_total))
    state_0_base[2, :] = 1.0 

    # Loop Principal: Evolução temporal para cada ponto de dessintonia
    for i, W0 in enumerate(W_0_range):
        delta = W0 + w_zeeman
        delta_flat = np.full(N_total, delta)
        
        # 1º Pulso: Subida pela cavidade
        Omega_vec_pulse_1 = np.array([Omega_local_flat_1, np.zeros(N_total), delta_flat])
        state_mid = apply_bloch_rotation(state_0_base, Omega_vec_pulse_1, t_local_flat)
        
        # Voo livre: Evolução de fase na região sem micro-ondas (apenas delta atua)
        Omega_vec_free = np.array([np.zeros(N_total), np.zeros(N_total), delta_flat])
        state_evolved = apply_bloch_rotation(state_mid, Omega_vec_free, T_local_flat)
        
        # 2º Pulso: Descida pela cavidade
        Omega_vec_pulse_2 = np.array([Omega_local_flat_2, np.zeros(N_total), delta_flat])
        state_final = apply_bloch_rotation(state_evolved, Omega_vec_pulse_2, t_local_flat)
        
        # Probabilidade P2 de estado excitado ponderada pela distribuição de Maxwell-Boltzmann
        P2_local = 0.5 * (1 - state_final[2, :])
        P2[i] = np.sum(P2_local * Weights_flat)

    return tvec, P2

# Bloco executado apenas se o arquivo for chamado diretamente no terminal
if __name__ == "__main__":
    print("Executando simulação para geração de CSV...")
    
    # Parâmetros padrão para a exportação
    T0_init = 1.0
    T_init = 0.22
    t_init = 1.7e-2
    B_fixed = 3.0e-9

    tvec, P2 = simulate_ramsey_fringe(T0_init * 1e-6, T_init, t_init, B_fixed)
    
    # Geração do arquivo estruturado com cabeçalho
    arquivo_csv = 'data/simulation_data.csv'
    np.savetxt(arquivo_csv, np.column_stack((tvec, P2)), delimiter=',', 
               header='Detuning_Normalizado,Probabilidade_P2', comments='')
    
    print(f"Simulação concluída. Resultados salvos em '{arquivo_csv}'.")
