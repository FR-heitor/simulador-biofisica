# ==========================================
# DEPENDÊNCIAS NECESSÁRIAS (Instale no terminal):
# pip install streamlit numpy scipy matplotlib
# ==========================================

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Simulador de Biofísica Completo", layout="wide")

# --- CONSTANTES FÍSICAS ---
R = 8.314  # Constante dos gases (J/(mol·K))
T = 310    # Temperatura absoluta (K) (37°C)
F = 96485  # Constante de Faraday (C/mol)

# --- FUNÇÕES MATEMÁTICAS ESTÁTICAS ---
def calcular_nernst(valence, c_in, c_out):
    return ((R * T) / (valence * F)) * np.log(c_out / c_in) * 1000

def calcular_ghk(Na_i, Na_e, P_Na, K_i, K_e, P_K, Cl_i, Cl_e, P_Cl):
    num = (P_K * K_e) + (P_Na * Na_e) + (P_Cl * Cl_i)
    den = (P_K * K_i) + (P_Na * Na_i) + (P_Cl * Cl_e)
    if den <= 0: return 0
    return ((R * T) / F) * np.log(num / den) * 1000

# --- BARRA LATERAL (CONTROLES INTERATIVOS CÉLULA) ---
st.sidebar.title("🎛️ Nível Celular (Íons)")
st.sidebar.markdown("Parâmetros para as Abas 1, 2 e 3.")

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

# Cálculos Estáticos Iniciais
E_Na = calcular_nernst(1, Na_i, Na_e)
E_K = calcular_nernst(1, K_i, K_e)
E_Cl = calcular_nernst(-1, Cl_i, Cl_e)
Vm = calcular_ghk(Na_i, Na_e, P_Na, K_i, K_e, P_K, Cl_i, Cl_e, P_Cl)

# --- CORPO PRINCIPAL ---
st.title("🔬 Plataforma Integrada de Biofísica")

# Adicionamos a 4ª aba
aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Equilíbrio (GHK)", 
    "📈 Simulação (Não-Excitável)", 
    "⚡ Excitáveis (PA)",
    "🫀 Sinais Macroscópicos (ECG/EEG)"
])

# ==========================================================
# ABA 1: POTENCIAIS DE EQUILÍBRIO
# ==========================================================
with aba1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Valores Calculados")
        st.info(f"**Potencial de Membrana em Repouso (GHK):** {Vm:.2f} mV")
        st.success(f"**E_Na (Nernst):** {E_Na:.2f} mV")
        st.warning(f"**E_K (Nernst):** {E_K:.2f} mV")
        st.error(f"**E_Cl (Nernst):** {E_Cl:.2f} mV")
        
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

# ==========================================================
# ABA 2: SIMULAÇÃO TEMPORAL (CÉLULA NÃO EXCITÁVEL)
# ==========================================================
with aba2:
    st.subheader("Dinâmica Iônica Base")
    tempo_simulacao = st.slider("Duração da simulação (s)", 50, 500, 200, key="sim1")
    
    if st.button("▶️ Rodar Simulação", key="btn1"):
        def modelo(t, y):
            Na_in, K_in, Cl_in = y
            dNa = -P_Na * (Na_in - Na_e) * 0.01
            dK = -P_K * (K_in - K_e) * 0.01
            dCl = -P_Cl * (Cl_in - Cl_e) * 0.01
            return [dNa, dK, dCl]

        t_eval = np.linspace(0, tempo_simulacao, 200)
        sol = solve_ivp(modelo, (0, tempo_simulacao), [Na_i, K_i, Cl_i], t_eval=t_eval)

        vms = [calcular_ghk(sol.y[0][i], Na_e, P_Na, sol.y[1][i], K_e, P_K, sol.y[2][i], Cl_e, P_Cl) for i in range(len(sol.t))]

        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(sol.t, vms, 'b-', label='Vm (GHK)', lw=2)
        ax1.set_ylabel('Potencial (mV)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(sol.t, sol.y[0], 'g-', label='[Na+]i', lw=2)
        ax2.plot(sol.t, sol.y[1], 'orange', label='[K+]i', lw=2)
        ax2.plot(sol.t, sol.y[2], 'r-', label='[Cl-]i', lw=2)
        ax2.set_xlabel('Tempo (s)')
        ax2.set_ylabel('Concentração (mM)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)

# ==========================================================
# ABA 3: CÉLULAS EXCITÁVEIS (HODGKIN-HUXLEY & MÚSCULOS)
# ==========================================================
with aba3:
    st.subheader("⚡ Eletrofisiologia de Células Excitáveis")
    st.markdown("Análise temporal detalhada de Potenciais de Ação, Condutâncias e Correntes Iônicas.")
    
    tipo_celula = st.radio(
        "Selecione o Tecido para Estudo:", 
        ["Neurônio (Modelo Hodgkin-Huxley)", "Músculo Esquelético", "Músculo Cardíaco (Ventricular)"],
        horizontal=True
    )

    st.markdown("---")
    
    col_est1, col_est2 = st.columns(2)
    with col_est1:
        estimulo = st.slider("Corrente de Estímulo (µA/cm²)", 0.0, 50.0, 15.0)
    with col_est2:
        if tipo_celula == "Músculo Cardíaco (Ventricular)":
            tempo_maximo = st.slider("Tempo de Observação (ms)", 100, 600, 400)
        elif tipo_celula == "Músculo Esquelético":
            tempo_maximo = st.slider("Tempo de Observação (ms)", 10, 100, 30)
        else:
            tempo_maximo = st.slider("Tempo de Observação (ms)", 10, 100, 50)

    if st.button("⚡ Aplicar Estímulo e Simular"):
        
        # 1. MODELO DE HODGKIN-HUXLEY (NEURÔNIO)
        if tipo_celula == "Neurônio (Modelo Hodgkin-Huxley)":
            # Parâmetros clássicos do axônio gigante da lula
            C_m = 1.0      # µF/cm²
            g_Na_max = 120.0 # mS/cm²
            g_K_max = 36.0   # mS/cm²
            g_L = 0.3        # mS/cm²
            E_Na = 50.0      # mV
            E_K = -77.0      # mV
            E_L = -54.387    # mV

            # Funções de transição (Alphas e Betas)
            def a_m(V): return 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0)) if V != -40.0 else 1.0
            def b_m(V): return 4.0 * np.exp(-(V + 65.0) / 18.0)
            def a_h(V): return 0.07 * np.exp(-(V + 65.0) / 20.0)
            def b_h(V): return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
            def a_n(V): return 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0)) if V != -55.0 else 0.1
            def b_n(V): return 0.125 * np.exp(-(V + 65.0) / 80.0)

            def hodgkin_huxley(t, y):
                V, m, h, n = y
                I_inj = estimulo if 5.0 <= t <= 6.0 else 0.0 # Estímulo de 1ms
                
                I_Na = g_Na_max * (m**3) * h * (V - E_Na)
                I_K = g_K_max * (n**4) * (V - E_K)
                I_L = g_L * (V - E_L)
                
                dVdt = (I_inj - I_Na - I_K - I_L) / C_m
                dmdt = a_m(V) * (1.0 - m) - b_m(V) * m
                dhdt = a_h(V) * (1.0 - h) - b_h(V) * h
                dndt = a_n(V) * (1.0 - n) - b_n(V) * n
                return [dVdt, dmdt, dhdt, dndt]

            # Condições iniciais no repouso (-65 mV)
            V0 = -65.0
            m0 = a_m(V0) / (a_m(V0) + b_m(V0))
            h0 = a_h(V0) / (a_h(V0) + b_h(V0))
            n0 = a_n(V0) / (a_n(V0) + b_n(V0))

            t_eval = np.linspace(0, tempo_maximo, 1000)
            sol = solve_ivp(hodgkin_huxley, [0, tempo_maximo], [V0, m0, h0, n0], t_eval=t_eval, method='RK45')

            # Recuperando as variáveis resolvidas
            t = sol.t
            V = sol.y[0]
            m = sol.y[1]
            h = sol.y[2]
            n = sol.y[3]

            # Calculando Condutâncias e Correntes
            g_Na = g_Na_max * (m**3) * h
            g_K = g_K_max * (n**4)
            I_Na = g_Na * (V - E_Na)
            I_K = g_K * (V - E_K)

            # --- PLOTAGEM NEURÔNIO (4 GRÁFICOS) ---
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
            
            # 1. Potencial de Membrana
            ax1.plot(t, V, 'purple', lw=2)
            ax1.axvspan(5, 6, color='yellow', alpha=0.3, label='Estímulo')
            ax1.set_ylabel('Vm (mV)')
            ax1.set_title('Modelo Hodgkin-Huxley: Potencial de Membrana')
            ax1.grid(alpha=0.3)
            ax1.legend()

            # 2. Correntes
            ax2.plot(t, I_Na, 'green', lw=2, label='I_Na (Sódio)')
            ax2.plot(t, I_K, 'orange', lw=2, label='I_K (Potássio)')
            ax2.axhline(0, color='black', lw=1, ls='--')
            ax2.set_ylabel('Corrente (µA/cm²)')
            ax2.set_title('Correntes Iônicas Transmembrana')
            ax2.grid(alpha=0.3)
            ax2.legend()

            # 3. Condutâncias
            ax3.plot(t, g_Na, 'green', lw=2, ls='--', label='g_Na')
            ax3.plot(t, g_K, 'orange', lw=2, ls='--', label='g_K')
            ax3.set_ylabel('Condutância (mS/cm²)')
            ax3.set_title('Condutâncias Macroscópicas')
            ax3.grid(alpha=0.3)
            ax3.legend()

            # 4. Subpartículas (m, h, n)
            ax4.plot(t, m, 'g-', lw=2, label='m (Ativação Na+)')
            ax4.plot(t, h, 'r-', lw=2, label='h (Inativação Na+)')
            ax4.plot(t, n, 'orange', lw=2, label='n (Ativação K+)')
            ax4.set_ylabel('Probabilidade (0 a 1)')
            ax4.set_xlabel('Tempo (ms)')
            ax4.set_title('Dinâmica das Subpartículas (Gating Variables)')
            ax4.grid(alpha=0.3)
            ax4.legend()

            plt.tight_layout()
            st.pyplot(fig)


        # 2. MODELOS MUSCULARES (FENOMENOLÓGICOS COM CONDUTÂNCIA)
        else:
            dt = 0.1
            t = np.arange(0, tempo_maximo, dt)
            V = np.full(len(t), -90.0)
            g_Na = np.zeros(len(t))
            g_K = np.full(len(t), 0.5) # Repouso K
            g_Ca = np.zeros(len(t))
            
            fase = 0
            t_fase = 0

            for i in range(1, len(t)):
                # Estímulo
                if 5.0 <= t[i] <= 6.0 and fase == 0:
                    V[i] = V[i-1] + (estimulo * dt * 5)
                else:
                    V[i] = V[i-1]

                if tipo_celula == "Músculo Esquelético":
                    if V[i] > -55.0 and fase == 0: fase = 1
                    
                    if fase == 1: # Spike Na+
                        g_Na[i] = 30.0
                        V[i] = min(V[i-1] + 1200 * dt, 30.0)
                        if V[i] >= 29.0: fase = 2
                    elif fase == 2: # Repolarização K+
                        g_K[i] = 15.0
                        V[i] = max(V[i-1] - 400 * dt, -90.0)
                        if V[i] <= -90.0: fase = 3
                    else:
                        V[i] = max(V[i-1] - 80 * dt, -90.0)

                elif tipo_celula == "Músculo Cardíaco (Ventricular)":
                    if V[i] > -40.0 and fase == 0: fase = 1
                    
                    if fase == 1: # Fase 0: Na+
                        g_Na[i] = 20.0
                        V[i] = min(V[i-1] + 500 * dt, 20.0)
                        if V[i] >= 19.0: fase = 2
                    elif fase == 2: # Fase 1: Notch
                        g_K[i] = 5.0
                        V[i] = max(V[i-1] - 100 * dt, 5.0)
                        if V[i] <= 6.0: 
                            fase = 3
                            t_fase = t[i]
                    elif fase == 3: # Fase 2: Platô (Ca2+)
                        g_Ca[i] = 8.0  # Influxo de Cálcio
                        g_K[i] = 2.0   # Efluxo reduzido de K
                        V[i] = V[i-1] - 0.05 * dt
                        if t[i] - t_fase > 200: fase = 4
                    elif fase == 4: # Fase 3: Repolarização
                        g_K[i] = 10.0
                        V[i] = max(V[i-1] - 150 * dt, -90.0)
                    else:
                        V[i] = max(V[i-1] - 30 * dt, -90.0)

            # Calculando correntes simuladas com base nas condutâncias (I = g*(V - E))
            I_Na = g_Na * (V - 50.0)
            I_K = g_K * (V - (-90.0))
            I_Ca = g_Ca * (V - 120.0) if tipo_celula == "Músculo Cardíaco (Ventricular)" else np.zeros(len(t))

            # --- PLOTAGEM MUSCULAR (3 GRÁFICOS) ---
            fig2, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            cor = 'green' if "Esquelético" in tipo_celula else 'red'

            # 1. Voltagem
            ax1.plot(t, V, color=cor, lw=2)
            ax1.axvspan(5, 6, color='yellow', alpha=0.3)
            ax1.set_ylabel('Vm (mV)')
            ax1.set_title(f'Potencial de Membrana: {tipo_celula}')
            ax1.grid(alpha=0.3)

            # 2. Correntes
            ax2.plot(t, I_Na, 'green', label='I_Na')
            ax2.plot(t, I_K, 'orange', label='I_K')
            if "Cardíaco" in tipo_celula:
                ax2.plot(t, I_Ca, 'purple', label='I_Ca (Cálcio tipo L)')
            ax2.axhline(0, color='black', lw=1, ls='--')
            ax2.set_ylabel('Corrente Estimada')
            ax2.set_title('Correntes Iônicas (Simulação Fenomenológica)')
            ax2.legend()
            ax2.grid(alpha=0.3)

            # 3. Condutâncias
            ax3.plot(t, g_Na, 'green', ls='--', label='g_Na')
            ax3.plot(t, g_K, 'orange', ls='--', label='g_K')
            if "Cardíaco" in tipo_celula:
                ax3.plot(t, g_Ca, 'purple', ls='--', label='g_Ca')
            ax3.set_ylabel('Condutância Relativa')
            ax3.set_xlabel('Tempo (ms)')
            ax3.set_title('Condutâncias Macroscópicas')
            ax3.legend()
            ax3.grid(alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig2)

# ==========================================================
# ABA 4: SINAIS MACROSCÓPICOS (ECG e EEG) - NOVA!
# ==========================================================
with aba4:
    st.subheader("🫀 Eletrofisiologia Macroscópica")
    st.markdown("A soma das atividades elétricas individuais gera assinaturas mensuráveis em órgãos e tecidos.")
    
    tipo_sinal = st.radio("Selecione o Exame Simulado:", ["Eletrocardiograma (ECG)", "Eletroencefalograma (EEG)"], horizontal=True)
    st.markdown("---")
    
    # ------------------ LÓGICA DO ECG ------------------
    if tipo_sinal == "Eletrocardiograma (ECG)":
        col_ecg1, col_ecg2 = st.columns(2)
        with col_ecg1:
            bpm = st.slider("Frequência Cardíaca (BPM)", 40, 180, 75, help="Batimentos por minuto")
        with col_ecg2:
            ruido_ecg = st.slider("Artefato/Ruído Muscular (%)", 0.0, 50.0, 5.0) / 100.0

        if st.button("Gerar Traçado ECG"):
            fs = 500 # Taxa de amostragem
            t_ecg = np.linspace(0, 5, 5 * fs) # 5 segundos de gravação
            
            # Cálculo do intervalo entre batimentos
            rr_interval = 60.0 / bpm
            
            # Função para gerar um ciclo cardíaco padrão (P, Q, R, S, T) usando funções Gaussianas
            def pulso_gaussiano(t, pico, largura, centro):
                return pico * np.exp(-((t - centro) ** 2) / (2 * largura ** 2))
            
            ecg_signal = np.zeros_like(t_ecg)
            
            for i, t in enumerate(t_ecg):
                t_ciclo = t % rr_interval # Tempo dentro do batimento atual
                
                # Modelagem das ondas
                onda_p = pulso_gaussiano(t_ciclo, pico=0.25, largura=0.015, centro=0.2)
                onda_q = pulso_gaussiano(t_ciclo, pico=-0.15, largura=0.01, centro=0.35)
                onda_r = pulso_gaussiano(t_ciclo, pico=1.2, largura=0.015, centro=0.38)
                onda_s = pulso_gaussiano(t_ciclo, pico=-0.25, largura=0.015, centro=0.41)
                onda_t = pulso_gaussiano(t_ciclo, pico=0.35, largura=0.03, centro=0.6)
                
                ecg_signal[i] = onda_p + onda_q + onda_r + onda_s + onda_t
            
            # Adicionar ruído
            ecg_signal += np.random.normal(0, ruido_ecg, len(t_ecg))

            fig_ecg, ax_ecg = plt.subplots(figsize=(12, 4))
            ax_ecg.plot(t_ecg, ecg_signal, color='#ff3333', lw=1.5)
            ax_ecg.set_title(f'Eletrocardiograma Simulado ({bpm} BPM)')
            ax_ecg.set_xlabel('Tempo (s)')
            ax_ecg.set_ylabel('Amplitude (mV)')
            
            # Grade estilo papel de ECG clássico
            ax_ecg.grid(True, which='both', color='red', linestyle='-', alpha=0.2)
            ax_ecg.minorticks_on()
            ax_ecg.grid(True, which='minor', color='red', linestyle='-', alpha=0.1)
            
            st.pyplot(fig_ecg)
            
            st.info("💡 **Conceito:** A onda **P** representa a despolarização atrial. O complexo **QRS** marca a despolarização ventricular (e o mascaramento da repolarização atrial). A onda **T** sinaliza a repolarização ventricular.")

    # ------------------ LÓGICA DO EEG ------------------
    else:
        st.markdown("Ajuste as amplitudes das diferentes faixas de frequência (ritmos cerebrais):")
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: delta = st.slider("Delta (1-4 Hz)", 0.0, 5.0, 1.0, help="Sono profundo")
        with c2: theta = st.slider("Theta (4-8 Hz)", 0.0, 5.0, 0.5, help="Sonolência/Meditação")
        with c3: alfa = st.slider("Alfa (8-13 Hz)", 0.0, 5.0, 2.5, help="Relaxado, olhos fechados")
        with c4: beta = st.slider("Beta (13-30 Hz)", 0.0, 5.0, 1.0, help="Acordado, focado")
        with c5: gama = st.slider("Gama (30-100 Hz)", 0.0, 5.0, 0.2, help="Processamento cognitivo")
        
        if st.button("Gerar Traçado EEG"):
            fs_eeg = 250 # Amostragem EEG
            t_eeg = np.linspace(0, 3, 3 * fs_eeg) # 3 segundos
            
            # Geração de ondas misturando frequências aleatórias dentro das bandas
            def gerar_banda(amp, f_min, f_max, num_ondas=3):
                sinal = np.zeros_like(t_eeg)
                for _ in range(num_ondas):
                    f = np.random.uniform(f_min, f_max)
                    fase = np.random.uniform(0, 2*np.pi)
                    sinal += (amp / num_ondas) * np.sin(2 * np.pi * f * t_eeg + fase)
                return sinal
            
            sinal_delta = gerar_banda(delta, 1, 4)
            sinal_theta = gerar_banda(theta, 4, 8)
            sinal_alfa = gerar_banda(alfa, 8, 13)
            sinal_beta = gerar_banda(beta, 13, 30)
            sinal_gama = gerar_banda(gama, 30, 60)
            
            # Sinal final = soma + ruído base de fundo
            eeg_signal = sinal_delta + sinal_theta + sinal_alfa + sinal_beta + sinal_gama
            eeg_signal += np.random.normal(0, 0.2, len(t_eeg)) # Pequeno ruído elétrico/térmico

            # Realizar a Transformada de Fourier (FFT) para análise espectral
            fft_vals = np.abs(np.fft.rfft(eeg_signal))
            fft_freq = np.fft.rfftfreq(len(t_eeg), d=1/fs_eeg)

            # Plots
            fig_eeg, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Traçado do Tempo
            ax_t.plot(t_eeg, eeg_signal, color='black', lw=1.2)
            ax_t.set_title('Traçado EEG Bruto (3 Segundos)')
            ax_t.set_xlabel('Tempo (s)')
            ax_t.set_ylabel('Amplitude (µV)')
            ax_t.grid(alpha=0.3)
            
            # Espectro de Frequência (Análise Quantitativa)
            # Focamos até 40Hz para melhor visualização didática
            idx_limite = np.where(fft_freq <= 40)[0] 
            ax_f.plot(fft_freq[idx_limite], fft_vals[idx_limite], color='blue')
            ax_f.fill_between(fft_freq[idx_limite], fft_vals[idx_limite], color='blue', alpha=0.3)
            
            # Marcadores de Banda
            ax_f.axvspan(1, 4, color='red', alpha=0.1, label='Delta')
            ax_f.axvspan(4, 8, color='orange', alpha=0.1, label='Theta')
            ax_f.axvspan(8, 13, color='green', alpha=0.1, label='Alfa')
            ax_f.axvspan(13, 30, color='purple', alpha=0.1, label='Beta')
            
            ax_f.set_title('Espectro de Frequência (Transformada Rápida de Fourier - FFT)')
            ax_f.set_xlabel('Frequência (Hz)')
            ax_f.set_ylabel('Potência Relativa')
            ax_f.legend()
            ax_f.grid(alpha=0.3)
            
            st.pyplot(fig_eeg)
            
            st.success("🧠 **Dica Didática:** Observe como as alterações nos *sliders* afetam o padrão bruto (em cima) e deslocam os picos de frequência na análise FFT (embaixo). Isso ilustra como sistemas de neurofeedback e interfaces cérebro-máquina interpretam nosso estado mental!")
