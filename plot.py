import pandas as pd
import matplotlib.pyplot as plt

# Carrega os dados dos dois arquivos CSV
df1 = pd.read_csv('simulation_data.csv')
df2 = pd.read_csv('simulation_data_alt.csv')

# Criação da figura
plt.figure(figsize=(10, 6))

# Plota a primeira curva (simulation_data.csv)
plt.plot(df1['Detuning_Normalizado'], df1['Probabilidade_P2'], 
         color='blue', linewidth=2.0, label='Dados - Padrão')

# Plota a segunda curva (simulation_data_alt.csv)
plt.plot(df2['Detuning_Normalizado'], df2['Probabilidade_P2'], 
         color='red', linewidth=2.0, linestyle='--', label='Dados - Alternativo')

# Configuração visual do gráfico
plt.title('Franjas de Ramsey no Chafariz de Átomos Frios', fontsize=14)
plt.xlabel('Detuning Normalizado ($\\Delta \\cdot \\tau$)', fontsize=12)
plt.ylabel('Probabilidade de Transição ($P_2$)', fontsize=12)
plt.ylim(-0.05, 1.05)
plt.xlim(-20, 20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

# Ajusta o layout e exibe o gráfico
plt.tight_layout()
plt.show()