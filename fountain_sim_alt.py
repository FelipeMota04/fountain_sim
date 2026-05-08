import numpy as np
from scipy.special import j0

# =============================================================================
# 1. FUNÇÕES DO MODELO FÍSICO
# =============================================================================

def apply_bloch_rotation(state, Omega_vec, t):
    """
    Aplica rotação no vetor de Bloch usando a versão vetorial da Fórmula de Rodrigues.
    Otimizado para reduzir alocação de memória e operações de ponto flutuante.
    """
    # Norma da Frequência de Rabi
    W_norm = np.linalg.norm(Omega_vec, axis=0)
    W_norm = np.where(W_norm == 0, 1e-15, W_norm) # Evita divisão por zero
    
    # Componentes do eixo de rotação normalizado (vetor k)
    kx, ky, kz = Omega_vec / W_norm
    
    # Componentes do estado atual (vetor v)
    x, y, z = state
    
    # Ângulos de precessão
    theta = W_norm * t
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    inv_cos = 1.0 - cos_t
    
    # Produto vetorial (k x v)
    cross_x = ky * z - kz * y
    cross_y = kz * x - kx * z
    cross_z = kx * y - ky * x
    
    # Produto escalar (k . v)
    dot_kv = kx * x + ky * y + kz * z
    
    # Cálculo do novo estado: v*cos(t) + (k x v)*sin(t) + k*(k.v)*(1 - cos(t))
    term3 = dot_kv * inv_cos
    
    new_x = x * cos_t + cross_x * sin_t + kx * term3
    new_y = y * cos_t + cross_y * sin_t + ky * term3
    new_z = z * cos_t + cross_z * sin_t + kz * term3
    
    return np.array([new_x, new_y, new_z])

def apply_bloch_rotation_XZ(state, Omega_x, Omega_z, t):
    """
    Rotação otimizada assumindo que o campo micro-ondas atua apenas no eixo X 
    e a dessintonia atua no eixo Z (componente Y nula).
    """
    # Norma calculada com operação elementar simples
    W_norm = np.sqrt(Omega_x**2 + Omega_z**2)
    W_norm = np.where(W_norm == 0, 1e-15, W_norm)
    
    kx = Omega_x / W_norm
    kz = Omega_z / W_norm
    
    x, y, z = state
    
    theta = W_norm * t
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    inv_cos = 1.0 - cos_t
    
    # Produto vetorial simplificado (ky = 0)
    cross_x = - kz * y
    cross_y = kz * x - kx * z
    cross_z = kx * y
    
    # Produto escalar simplificado (ky = 0)
    dot_kv = kx * x + kz * z
    term3 = dot_kv * inv_cos
    
    new_x = x * cos_t + cross_x * sin_t + kx * term3
    new_y = y * cos_t + cross_y * sin_t  # termo ky foi eliminado
    new_z = z * cos_t + cross_z * sin_t + kz * term3
    
    return np.array([new_x, new_y, new_z])


def apply_shaped_pulse(state, Omega_amp, delta_flat, t_local, profile_type='sine', N_steps=30):
    """
    Integra a evolução do vetor de Bloch de forma vetorizada usando a rotação XZ.
    """
    current_state = np.copy(state)
    tau_edges = np.linspace(0, 1, N_steps + 1)
    tau_mid = (tau_edges[:-1] + tau_edges[1:]) / 2.0
    dtau = 1.0 / N_steps
    dt_local = t_local * dtau
    
    for tau in tau_mid:
        if profile_type == 'sine':
            envelope = (np.pi / 2) * np.sin(np.pi * tau)
        elif profile_type == 'blackman':
            envelope = 0.42 - 0.5 * np.cos(2 * np.pi * tau) + 0.08 * np.cos(4 * np.pi * tau)
            envelope *= (1.0 / 0.42)
        elif profile_type == "square": # square
            envelope = 1.0
            
        # O envelope atua apenas na amplitude base
        Omega_x = Omega_amp * envelope 
        
        # Chama a função otimizada que recebe x e z diretamente
        current_state = apply_bloch_rotation_XZ(current_state, Omega_x, delta_flat, dt_local)
        
    return current_state


def simulate_ramsey_fringe(Temp_0, T, t, B=3e-9,theta1=np.pi/2,
                           theta2 = np.pi/2, N_r=50, N_v=30,
                           profile1='sine', profile2='sine'):
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
    
    # Transforma W_0_range em uma malha bidimensional
    delta_2D = (W_0_range + w_zeeman)[np.newaxis, :] 
    delta_flat = np.broadcast_to(delta_2D, (N_total, points))
    
    # Expande as dimensões dos parâmetros espaciais e cinemáticos
    Omega_1_2D = Omega_local_flat_1[:, np.newaxis]
    Omega_2_2D = Omega_local_flat_2[:, np.newaxis]
    
    t_local_2D = (t * (1 - V_flat / v_launch))[:, np.newaxis]
    T_local_2D = (T * (1 + V_flat / v_launch))[:, np.newaxis]

    # Estado quântico inicial 3D: (Componentes x/y/z, Átomos, Frequências)
    state_0_base = np.zeros((3, N_total, points))
    state_0_base[2, :, :] = 1.0 
    
    # 1º Pulso: Subida pela cavidade
    state_mid = apply_shaped_pulse(state_0_base, Omega_1_2D, delta_flat, 
                                   t_local_2D, profile_type=profile1, N_steps=30)
    
    # Voo livre: Evolução de fase na região escura (nâo calculamos a fórmula de rodrigues novamente para otimizar)
    
    phi = delta_flat * T_local_2D
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    x_mid, y_mid, z_mid = state_mid
    x_ev = x_mid * cos_phi - y_mid * sin_phi
    y_ev = y_mid * cos_phi + x_mid * sin_phi
    state_evolved = np.array([x_ev, y_ev, z_mid])
    
    # 2º Pulso: Descida pela cavidade
    state_final = apply_shaped_pulse(state_evolved, Omega_2_2D, delta_flat, 
                                     t_local_2D, profile_type=profile2, N_steps=30)
    
    # Probabilidade P2 ponderada. A integração ocorre apenas no eixo dos átomos (axis=0)
    P2_local = 0.5 * (1 - state_final[2, :, :])
    P2 = np.sum(P2_local * Weights_flat[:, np.newaxis], axis=0)

    return tvec, P2

# Bloco executado apenas se o arquivo for chamado diretamente no terminal
if __name__ == "__main__":
    print("Executando simulação para geração de CSV...")
    
    # Parâmetros padrão para a exportação
    T0_init = 16.0
    T_init = 0.1
    t_init = 1.0e-2
    B_fixed = 3.0e-9

    tvec, P2 = simulate_ramsey_fringe(T0_init * 1e-6, T_init, t_init, B_fixed,profile1='sine', profile2='sine')
    
    # Geração do arquivo estruturado com cabeçalho
    arquivo_csv = 'simulation_data_alt.csv'
    np.savetxt(arquivo_csv, np.column_stack((tvec, P2)), delimiter=',', 
               header='Detuning_Normalizado,Probabilidade_P2', comments='')
    
    print(f"Simulação concluída. Resultados salvos em '{arquivo_csv}'.")