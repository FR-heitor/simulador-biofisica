# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:30:39 2026

@author: franc
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Simulador de Biofísica", layout="wide")

# --- CONSTANTES FÍSICAS ---
R = 8.314  # Constante dos gases (J/(mol·K))
T = 310    # Temperatura absoluta (K) (37°C)
F = 96485  # Constante de Faraday (C/mol)

# --- FUNÇÕES MATEMÁTICAS ---
def calcular_nernst(valence, c_in, c_out):
    # E = (RT/zF) * ln(C_out / C_in)
    return ((R * T) / (valence * F)) * np.log(c_out / c_in) * 1000

def calcular_ghk(Na_i, Na_e, P_Na, K_i, K_e, P_K, Cl_i, Cl_e, P_Cl):
    num = (P_K * K_e) + (P_Na * Na_e) + (P_Cl * Cl_i)
    den = (P_K * K_i) + (P_Na * Na_i) + (P_Cl * Cl_e)
    if den <= 0: return 0
    return ((R * T) / F) * np.log(num / den) * 1000

# --- BARRA LATERAL (CONTROLES INTERATIVOS) ---
st.sidebar.title("🎛️ Manipulação de Variáveis")
st.sidebar.markdown("Altere os valores abaixo e veja os gráficos se atualizarem em tempo real.")

st.sidebar.markdown("### Sódio (Na+)")
Na_i = st.sidebar.slider("[Na+] Intracelular (mM)", 1.0, 50.0, 15.0)
Na_e = st.sidebar.slider("[Na+] Extracelular (mM)", 50.0, 200.0, 145.0)
P_Na = st.sidebar.slider("Permeabilidade P_Na", 0.00, 5.0, 0.05)

st.sidebar.markdown("### Potássio (K+)")
K_i = st.sidebar.slider("[K+] Intracelular (mM)", 50.0, 200.0, 150.0)
K_e = st.sidebar.slider("[K+] Extracelular (mM)", 1.0, 20.0, 5.0)
P_K = st.sidebar.slider("Permeabilidade P_K", 0.00, 5.0, 1.0)

st.sidebar.markdown("### Cloreto (Cl-)")
Cl_i = st.sidebar.slider("[Cl-] Intracelular (mM)", 1.0, 50.0, 10.0)
Cl_e = st.sidebar.slider("[Cl-] Extracelular (mM)", 50.0, 200.0, 110.0)
P_Cl = st.sidebar.slider("Permeabilidade P_Cl", 0.00, 2.0, 0.45)

# --- CORPO PRINCIPAL ---
st.title("🔬 Plataforma Integrada de Biofísica")

# Cálculos Estáticos Iniciais
E_Na = calcular_nernst(1, Na_i, Na_e)
E_K = calcular_nernst(1, K_i, K_e)
E_Cl = calcular_nernst(-1, Cl_i, Cl_e)
Vm = calcular_ghk(Na_i, Na_e, P_Na, K_i, K_e, P_K, Cl_i, Cl_e, P_Cl)

# Abas de visualização
aba1, aba2 = st.tabs(["📊 Potenciais de Equilíbrio", "📈 Simulação Temporal e Correntes"])

with aba1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Valores Calculados (Tempo Real)")
        st.info(f"**Potencial de Membrana em Repouso (GHK):** {Vm:.2f} mV")
        st.success(f"**E_Na (Nernst):** {E_Na:.2f} mV")
        st.warning(f"**E_K (Nernst):** {E_K:.2f} mV")
        st.error(f"**E_Cl (Nernst):** {E_Cl:.2f} mV")
        
        st.markdown("---")
        st.markdown("**Força Eletromotriz (Driving Force):**")
        st.markdown(f"Na+: {Vm - E_Na:.2f} mV *(Diferença entre Vm e E_Na)*")
        st.markdown(f"K+: {Vm - E_K:.2f} mV *(Diferença entre Vm e E_K)*")
        st.markdown(f"Cl-: {Vm - E_Cl:.2f} mV *(Diferença entre Vm e E_Cl)*")

    with col2:
        st.subheader("Balanço Eletroquímico")
        fig, ax = plt.subplots(figsize=(6, 4))
        ions = ['E_K', 'Vm (GHK)', 'E_Cl', 'E_Na']
        valores = [E_K, Vm, E_Cl, E_Na]
        cores = ['orange', 'blue', 'red', 'green']
        
        ax.barh(ions, valores, color=cores)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_xlabel("Potencial (mV)")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)

with aba2:
    st.subheader("Dinâmica Iônica (Célula Não-Excitável)")
    tempo_simulacao = st.slider("Duração da simulação (segundos)", 50, 500, 200)
    
    if st.button("▶️ Rodar Simulação Temporal"):
        def modelo(t, y):
            Na_in, K_in, Cl_in = y
            dNa = -P_Na * (Na_in - Na_e) * 0.01
            dK = -P_K * (K_in - K_e) * 0.01
            dCl = -P_Cl * (Cl_in - Cl_e) * 0.01
            return [dNa, dK, dCl]

        t_eval = np.linspace(0, tempo_simulacao, 200)
        sol = solve_ivp(modelo, (0, tempo_simulacao), [Na_i, K_i, Cl_i], t_eval=t_eval)

        # Calculando Vm e Correntes ao longo do tempo
        vms = []
        i_na = []
        i_k = []
        i_cl = []

        for i in range(len(sol.t)):
            # Vm dinâmico
            v_inst = calcular_ghk(sol.y[0][i], Na_e, P_Na, sol.y[1][i], K_e, P_K, sol.y[2][i], Cl_e, P_Cl)
            vms.append(v_inst)
            
            # Nernst instantâneo
            e_na_inst = calcular_nernst(1, sol.y[0][i], Na_e)
            e_k_inst = calcular_nernst(1, sol.y[1][i], K_e)
            e_cl_inst = calcular_nernst(-1, sol.y[2][i], Cl_e)
            
            # Correntes (I = g * (Vm - E))
            i_na.append(P_Na * (v_inst - e_na_inst))
            i_k.append(P_K * (v_inst - e_k_inst))
            i_cl.append(P_Cl * (v_inst - e_cl_inst))

        # Criando os 3 gráficos
        fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
        
        # Gráfico 1: Potencial de Membrana
        ax1.plot(sol.t, vms, 'b-', label='Vm (GHK)', linewidth=2)
        ax1.set_ylabel('Potencial (mV)')
        ax1.set_title('Evolução do Potencial de Membrana')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Gráfico 2: Concentrações
        ax2.plot(sol.t, sol.y[0], 'g-', label='[Na+]i', linewidth=2)
        ax2.plot(sol.t, sol.y[1], 'orange', label='[K+]i', linewidth=2)
        ax2.plot(sol.t, sol.y[2], 'r-', label='[Cl-]i', linewidth=2)
        ax2.set_ylabel('Concentração (mM)')
        ax2.set_title('Evolução das Concentrações Intracelulares')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Gráfico 3: Correntes Iônicas
        ax3.plot(sol.t, i_na, 'g-', label='I_Na (Corrente de Sódio)', linewidth=2)
        ax3.plot(sol.t, i_k, 'orange', label='I_K (Corrente de Potássio)', linewidth=2)
        ax3.plot(sol.t, i_cl, 'r-', label='I_Cl (Corrente de Cloreto)', linewidth=2)
        ax3.axhline(0, color='black', linewidth=1, linestyle='--')
        ax3.set_xlabel('Tempo (s)')
        ax3.set_ylabel('Corrente Estimada')
        ax3.set_title('Dinâmica das Correntes Iônicas')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig2)