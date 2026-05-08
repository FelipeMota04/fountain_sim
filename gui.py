import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from fountain_sim import simulate_ramsey_fringe

# =============================================================================
# 2. INTERFACE GRÁFICA (GUI) E PLOTAGEM
# =============================================================================

# Parâmetros iniciais da simulação
T0_init = 16.0     # Temperatura (µK)
T_init = 0.8       # Tempo de voo livre (s)
t_init = 1.0e-2    # Tempo na cavidade (s)
B_fixed = 3.0e-9   # Campo magnético fixo em 3 nT
theta1_init = 0.5  # Pulso na 1a cavidade em unidades de pi
theta2_init = 0.5  # Pulso na 2a cavidade em unidades de pi

# Configuração da figura e área de plotagem
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.45) # Reserva espaço inferior para os controles deslizantes

# Realiza a primeira simulação para preencher o gráfico ao abrir a janela
tvec, P2_washing_out = simulate_ramsey_fringe(T0_init*1e-6, T_init, t_init,
                                               B_fixed, theta1_init*np.pi, theta2_init*np.pi)
line_wo, = ax.plot(tvec, P2_washing_out, color='blue', linewidth=2.0, label='Curva Simulada')

# Estilização visual do gráfico
ax.set_title('Franjas de Ramsey no Chafariz de Átomos Frios', fontsize=14)
ax.set_xlabel('Detuning Normalizado ($\\Delta \\cdot \\tau$)', fontsize=12)
ax.set_ylabel('Probabilidade de Transição ($P_2$)', fontsize=12)
ax.set_ylim(-0.05, 1.05)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper right')

# Caixa de texto centralizada para exibir variáveis calculadas dinamicamente
info_text = fig.text(0.5, 0.38, '', fontsize=11, ha='center', va='center', 
                     bbox=dict(facecolor='#f0f0f0', alpha=0.8, edgecolor='gray'))

# Definição dos eixos dos sliders: [esquerda, base, largura, altura]
ax_T0 = plt.axes([0.2, 0.30, 0.65, 0.025])
ax_T  = plt.axes([0.2, 0.24, 0.65, 0.025])
ax_t  = plt.axes([0.2, 0.18, 0.65, 0.025])
ax_th1  = plt.axes([0.2, 0.12, 0.65, 0.025])
ax_th2  = plt.axes([0.2, 0.06, 0.65, 0.025])



# Criação dos componentes interativos da interface
slider_T0 = Slider(ax_T0, 'Temp. (µK)', 0.1, 50.0, valinit=T0_init)
slider_T  = Slider(ax_T, 'Tempo de Vôo Livre (T) (s)', 0.01, 1.0, valinit=T_init)
slider_t  = Slider(ax_t, 'Tempo na Cavidade ($\\tau $)  (s)', 0.001, 0.1, valinit=t_init)
slider_th1  = Slider(ax_th1, 'Primeiro pulso (x $\\pi$)  (s)', 0.0, 2.0, valinit=theta1_init)
slider_th2  = Slider(ax_th2, 'Segundo pulso (x $\\pi$)  (s)', 0.0, 2.0, valinit=theta2_init)



def update(val):
    """
    Função de callback acionada na movimentação de qualquer slider.
    """
    # Coleta e converte os valores atuais selecionados
    T0_val = slider_T0.val * 1e-6
    T_val  = slider_T.val
    t_val  = slider_t.val
    th1_val = slider_th1.val * np.pi
    th2_val = slider_th2.val * np.pi
    
    # Recálculo das variáveis cinemáticas informativas
    g = 9.80665
    v_launch = 0.5 * g * (T_val + 2 * t_val)
    Omega_1 = th1_val / t_val
    Omega_2 = th2_val / t_val
    
    # Atualiza o display das variáveis no gráfico
    info_text.set_text(f'$V_z$ = {v_launch:.2f} m/s | $\\Omega_1$ = {Omega_1:.1f} rad/s | $\\Omega_2$ = {Omega_2:.1f} rad/s')

    # Refaz o cálculo de expansão de Monte Carlo com os novos parâmetros
    new_tvec, new_P2 = simulate_ramsey_fringe(T0_val, T_val, t_val, B_fixed, th1_val, th2_val)

    # Substitui os dados da linha sem a necessidade de recriar a figura (melhora a performance)
    line_wo.set_xdata(new_tvec)
    line_wo.set_ydata(new_P2)
    ax.set_xlim(-20, 20)
    
    # Força a interface gráfica a se redesenhar
    fig.canvas.draw_idle()

# Chamada manual da atualização na inicialização para preencher a caixa de texto
update(0)

# Associação do evento 'on_changed' de cada slider à função de callback
slider_T0.on_changed(update)
slider_T.on_changed(update)
slider_t.on_changed(update)
slider_th1.on_changed(update)
slider_th2.on_changed(update)

plt.show()