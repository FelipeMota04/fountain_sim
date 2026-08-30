# fountain_sim

Simulação física e ajuste das **franjas de Ramsey** de um **chafariz atômico** de
césio-133. O projeto inclui a simulação do vetor de Bloch, uma versão **otimizada** com
perfil não-instantâneo de ativação do campo de micro-ondas e um código de **ajuste** aos
dados experimentais.

## Física

Em um chafariz atômico, uma nuvem de átomos de Cs-133 é lançada verticalmente e atravessa
duas vezes uma cavidade de micro-ondas (na subida e na descida), realizando uma
interferometria de Ramsey. A probabilidade de transição `P2` oscila com a dessintonia
`Δν` entre o micro-ondas e a ressonância atômica, formando as franjas de Ramsey.

O ajuste infere parâmetros físicos como a **temperatura** da nuvem, o **tempo de voo
livre** `T`, o **tempo na cavidade** `τ` e o **campo de micro-ondas** (frequência de Rabi
`Ω = θ/τ`).

## Estrutura do projeto

```
fountain_sim/
├── fountain_sim.py               # simulação original (pulso quadrado)
├── fountain_sim_alt.py           # simulação vetorizada (pulso seno/Blackman)
├── fountain_sim_performance.py   # simulação OTIMIZADA (quadraturas + 7 perfis de pulso)
├── fit_data.py                   # ajuste aos dados experimentais (resposta instrumental)
├── gui.py                        # interface gráfica desktop interativa (matplotlib)
├── data/
│   ├── exp_data.csv              # dados experimentais (frequência × sinal)
│   ├── simulation_data.csv       # saída da simulação original
│   ├── simulation_data_alt.csv   # saída da simulação alternativa
│   └── simulacao_ramsey.csv      # curvas experimentais de referência
├── notebooks/
│   └── perfil_cavidade.ipynb     # análise do perfil da cavidade
├── images/                       # figuras geradas e imagens de referência
├── requirements.txt
└── LICENSE
```

## Instalação

As dependências são `numpy`, `scipy`, `pandas`, `matplotlib`, `streamlit` e `plotly`. Com `pip`:

pip install -r requirements.txt
```

## Uso

### Simular as franjas de Ramsey

```bash
python fountain_sim.py              # gera data/simulation_data.csv
python fountain_sim_alt.py          # gera data/simulation_data_alt.csv
```

### Ajustar a simulação aos dados experimentais

```bash
python fit_data.py                  # ajusta todos os perfis e gera images/fit_ramsey.png
```

O `fit_data.py`:

1. lê `data/exp_data.csv` e converte a frequência em dessintonia `Δν = ν − ν0`
   (`ν0 = 9 192 631 770 Hz`, transição do Cs-133);
2. modela a **resposta instrumental linear** `y = A·P2 + C`, com `A` (contraste) e
   `C` (fundo) perfilados analiticamente por mínimos quadrados;
3. ajusta os parâmetros físicos (temperatura, `T`, `τ`) com `θ = π/2` fixo;
4. repete o ajuste para sete perfis de ativação do micro-ondas
   (`square`, `sine`, `cos2`, `triangle`, `gaussian`, `blackman`, `gravity_sine`)
   e seleciona o melhor por `R²`/RMSE/AIC.

### Interface gráfica Desktop (Matplotlib)

```bash
python gui.py
```

### Dashboard Web Interativo (Streamlit)

```bash
streamlit run app.py
```

A modelagem da resposta instrumental elevou a qualidade do ajuste de
**R² ≈ 0,80 → 0,98** (RMSE ≈ 0,005 no sinal bruto), assentando corretamente os vales de
interferência destrutiva. A simulação otimizada (`fountain_sim_performance.py`) é
**9–14× mais rápida** que a versão vetorizada, com erro relativo **< 0,4%**.

O diagnóstico do ajuste indica que a temperatura converge para ~70 µK — acima do valor
físico de ~1 µK — sugerindo um mecanismo de *washing-out* não modelado (ex.: gradiente de
campo magnético, tamanho finito da nuvem).

## Licença

[MIT](LICENSE)
