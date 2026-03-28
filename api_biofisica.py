import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- CONFIGURAÇÃO DO AMBIENTE ---
st.set_page_config(page_title="Simulador de Biofísica UFS", layout="wide")

# --- CONSTANTES FÍSICAS UNIVERSAIS ---
R = 8.314  # Constante dos gases (J/(mol·K))
T = 310    # Temperatura (K) - 37°C
F = 96485  # Constante de Faraday (C/mol)

# --- FUNÇÕES MATEMÁTICAS ---
def calc_nernst(z, ci, co):
    """Calcula o Potencial de Equilíbrio de Nernst em mV."""
    if ci <= 0 or co <= 0: return 0
    return ((R * T) / (z * F)) * np.log(co / ci) * 1000

def calc_ghk(ni, ne, pn, ki, ke, pk, cli, cle, pcl):
    """Calcula o Potencial de Membrana de Goldman-Hodgkin-Katz em mV."""
    num = (pk * ke) + (pn * ne) + (pcl * cli)
    den = (pk * ki) + (pn * ni) + (pcl * cle)
    if den <= 0: return 0
    return ((R * T) / F) * np.log(num / den) * 1000

# --- BARRA LATERAL: PARÂMETROS E FARMACOLOGIA ---
st.sidebar.title("🧪 Laboratório de Biofísica")

st.sidebar.subheader("💊 Bloqueadores de Canais")
# Interface para simular fármacos bloqueadores
ttx = st.sidebar.toggle("Tetrodotoxina (TTX)", help="Bloqueia canais de Sódio (Na+). Causa paralisia.")
tea = st.sidebar.toggle("Tetraetilamónio (TEA)", help="Bloqueia canais de Potássio (K+).")
verapamil = st.sidebar.toggle("Verapamil", help="Bloqueia canais de Cálcio (Ca2+). Efeito Cardíaco.")

st.sidebar.divider()

st.sidebar.subheader("🧬 Concentrações e Permeabilidade")
# Ajuste de iões com impacto imediato no GHK
with st.sidebar.expander("Sódio (Na+)", expanded=True):
    na_i = st.slider("[Na+] Interno (mM)", 1.0, 50.0, 15.0)
    na_e = st.slider("[Na+] Externo (mM)", 50.0, 200.0, 145.0)
    p_na = st.slider("Permeabilidade P_Na", 0.0, 1.0, 0.02 if not ttx else 0.0)

with st.sidebar.expander("Potássio (K+)", expanded=True):
    k_i = st.slider("[K+] Interno (mM)", 50.0, 200.0, 150.0)
    k_e = st.slider("[K+] Externo (mM)", 1.0, 20.0, 5.0)
    p_k = st.slider("Permeabilidade P_K", 0.0, 2.0, 1.0 if not tea else 0.0)

with st.sidebar.expander("Cloreto (Cl-)", expanded=True):
    cl_i = st.slider("[Cl-] Interno (mM)", 1.0, 50.0, 10.0)
    cl_e = st.slider("[Cl-] Externo (mM)", 50.0, 200.0, 110.0)
    p_cl = st.slider("Permeabilidade P_Cl", 0.0, 1.0, 0.45)

# Cálculo do Potencial de Repouso Actual
vm_ghk = calc_ghk(na_i, na_e, p_na, k_i, k_e, p_k, cl_i, cl_e, p_cl)

# --- ECRÃ PRINCIPAL ---
st.title("🔬 Plataforma de Estudos Biofísicos")

tabs = st.tabs(["📊 Repouso (GHK)", "⚡ Células Excitáveis (PA)", "🫀 Sinais Macroscópicos"])

# ==========================================================
# ABA 1: POTENCIAL DE REPOUSO
# ==========================================================
with tabs[0]:
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Estado Eletroquímico")
        st.metric("Potencial de Membrana (Vm)", f"{vm_ghk:.2f} mV")
        st.write(f"E_Na: {calc_nernst(1, na_i, na_e):.1f} mV")
        st.write(f"E_K: {calc_nernst(1, k_i, k_e):.1f} mV")
        st.write(f"E_Cl: {calc_nernst(-1, cl_i, cl_e):.1f} mV")
    with c2:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(['K+', 'Vm (GHK)', 'Cl-', 'Na+'], 
                [calc_nernst(1, k_i, k_e), vm_ghk, calc_nernst(-1, cl_i, cl_e), calc_nernst(1, na_i, na_e)],
                color=['orange', 'blue', 'red', 'green'])
        ax.axvline(0, color='black', lw=1)
        ax.set_title("Equilíbrio de Nernst vs Realidade GHK")
        st.pyplot(fig)

# ==========================================================
# ABA 2: CÉLULAS EXCITÁVEIS (PA DETALHADO)
# ==========================================================
with tabs[1]:
    tipo_celula = st.radio("Selecione o Tecido:", ["Neurónio (Hodgkin-Huxley)", "Músculo Esquelético", "Músculo Cardíaco"], horizontal=True)
    
    st.info(f"O Repouso inicial é de {vm_ghk:.1f} mV (ajustável na barra lateral).")
    estimulo = st.slider("Intensidade do Estímulo Elétrico", 0.0, 50.0, 20.0)

    if st.button("⚡ Executar Simulação de Potencial de Ação"):
        
        # --- CASO 1: NEURÓNIO (HODGKIN-HUXLEY) ---
        if tipo_celula == "Neurónio (Hodgkin-Huxley)":
            def hh_model(t, y):
                V, m, h, n = y
                g_na = 120.0 if not ttx else 0.0
                g_k = 36.0 if not tea else 0.0
                g_l, e_na, e_k, e_l = 0.3, 50.0, -77.0, -54.4
                i_inj = estimulo if 5.0 <= t <= 6.0 else 0.0
                
                am = 0.1*(V+40)/(1-np.exp(-(V+40)/10)) if V != -40 else 1.0
                bm = 4.0*np.exp(-(V+65)/18)
                ah = 0.07*np.exp(-(V+65)/20)
                bh = 1.0/(1+np.exp(-(V+35)/10))
                an = 0.01*(V+55)/(1-np.exp(-(V+55)/10)) if V != -55 else 0.1
                bn = 0.125*np.exp(-(V+65)/80)
                
                dvdt = (i_inj - g_na*(m**3)*h*(V-e_na) - g_k*(n**4)*(V-e_k) - g_l*(V-e_l))
                return [dvdt, am*(1-m) - bm*m, ah*(1-h) - bh*h, an*(1-n) - bn*n]

            sol = solve_ivp(hh_model, [0, 50], [-65, 0.05, 0.6, 0.32], t_eval=np.linspace(0, 50, 1000))
            
            fig, ax = plt.subplots(4, 1, figsize=(10, 16), sharex=True)
            ax[0].plot(sol.t, sol.y[0], 'purple', lw=2); ax[0].set_ylabel("Vm (mV)"); ax[0].set_title("Potencial de Membrana")
            gna = 120 * (sol.y[1]**3) * sol.y[2] if not ttx else np.zeros_like(sol.t)
            gk = 36 * (sol.y[3]**4) if not tea else np.zeros_like(sol.t)
            ax[1].plot(sol.t, gna*(sol.y[0]-50), 'g', label="I_Na"); ax[1].plot(sol.t, gk*(sol.y[0]+77), 'orange', label="I_K")
            ax[1].set_ylabel("Corrente"); ax[1].legend()
            ax[2].plot(sol.t, gna, 'g--', label="g_Na"); ax[2].plot(sol.t, gk, 'orange', '--', label="g_K")
            ax[2].set_ylabel("Condutância"); ax[2].legend()
            ax[3].plot(sol.t, sol.y[1], 'g', label="m"); ax[3].plot(sol.t, sol.y[2], 'r', label="h"); ax[3].plot(sol.t, sol.y[3], 'orange', label="n")
            ax[3].set_ylabel("Gating"); ax[3].set_xlabel("Tempo (ms)"); ax[3].legend()
            st.pyplot(fig)

        # --- CASO 2: MÚSCULOS (MÁQUINA DE ESTADOS) ---
        else:
            t_max = 400 if tipo_celula == "Músculo Cardíaco" else 50
            t = np.linspace(0, t_max, 1000); dt = t[1] - t[0]
            v = np.full(1000, vm_ghk)
            gna, gk, gca = np.zeros(1000), np.zeros(1000), np.zeros(1000)
            fase = 0; t_f = 0

            for i in range(1, 1000):
                if 5.0 <= t[i] <= 7.0: v[i] = v[i-1] + (estimulo * dt * 5)
                else: v[i] = v[i-1]

                if tipo_celula == "Músculo Esquelético":
                    if v[i] > -55 and fase == 0 and not ttx: fase = 1
                    if fase == 1: 
                        gna[i] = 40; v[i] = min(v[i-1] + 1500*dt, 35)
                        if v[i] >= 34: fase = 2
                    elif fase == 2:
                        if not tea: gk[i] = 20; v[i] = max(v[i-1] - 500*dt, -90)
                        if v[i] <= -89: fase = 3
                    elif fase == 0: v[i] = max(v[i-1] - 50*dt, vm_ghk)
                
                else: # Cardíaco
                    if v[i] > -40 and fase == 0 and not ttx: fase = 1
                    if fase == 1:
                        gna[i] = 30; v[i] = min(v[i-1] + 600*dt, 20)
                        if v[i] >= 19: fase = 2
                    elif fase == 2:
                        v[i] = max(v[i-1] - 150*dt, 5); 
                        if v[i] <= 6: fase, t_f = 3, t[i]
                    elif fase == 3:
                        if not verapamil: gca[i] = 10; v[i] = v[i-1] - 0.05*dt
                        else: v[i] = v[i-1] - 80*dt 
                        if t[i] - t_f > 200 or verapamil: fase = 4
                    elif fase == 4:
                        if not tea: gk[i] = 15; v[i] = max(v[i-1] - 200*dt, -90)
                    elif fase == 0: v[i] = max(v[i-1] - 20*dt, vm_ghk)

            fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            ax[0].plot(t, v, 'red' if "Cardíaco" in tipo_celula else 'green', lw=2.5); ax[0].set_ylabel("Vm (mV)")
            ax[1].plot(t, gna*(v-50), 'g', label="I_Na"); ax[1].plot(t, gk*(v+90), 'orange', label="I_K")
            if "Cardíaco" in tipo_celula: ax[1].plot(t, gca*(v-120), 'purple', label="I_Ca")
            ax[1].set_ylabel("Corrente"); ax[1].legend()
            ax[2].plot(t, gna, 'g--', label="g_Na"); ax[2].plot(t, gk, 'orange', '--', label="g_K")
            if "Cardíaco" in tipo_celula: ax[2].plot(t, gca, 'purple', '--', label="g_Ca")
            ax[2].set_ylabel("Condutância"); ax[2].set_xlabel("Tempo (ms)"); ax[2].legend()
            st.pyplot(fig)

# ==========================================================
# ABA 3: SINAIS MACROSCÓPICOS (ECG / EEG / EMG-CMAP)
# ==========================================================
with tabs[2]:
    st.subheader("Simulação de Bio-sinais Macroscópicos")
    modo = st.radio("Selecione o Exame:", ["Eletrocardiograma (ECG)", "Eletroencefalograma (EEG)", "Eletromiograma (EMG / CMAP)"], horizontal=True)
    st.markdown("---")
    
    if modo == "Eletrocardiograma (ECG)":
        bpm = st.slider("Frequência Cardíaca (BPM)", 40, 180, 75)
        t_ecg = np.linspace(0, 5, 2500); rr = 60/bpm; ecg = np.zeros_like(t_ecg)
        for i, time in enumerate(t_ecg):
            tc = time % rr
            p = 0.2*np.exp(-((tc-0.2)**2)/(2*0.01**2))
            qrs = 1.2*np.exp(-((tc-0.4)**2)/(2*0.01**2))
            t_wave = 0.3*np.exp(-((tc-(0.7 if not verapamil else 0.5))**2)/(2*0.03**2))
            ecg[i] = p + qrs + t_wave
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t_ecg, ecg, 'r'); ax.set_title("Traçado ECG Simulado"); ax.set_xlabel("Tempo (s)"); ax.set_ylabel("mV")
        st.pyplot(fig)
    
    elif modo == "Eletroencefalograma (EEG)":
        st.write("Simulação de ritmos cerebrais compostos.")
        alfa = st.slider("Ritmo Alfa (Relaxamento)", 0.0, 5.0, 2.0)
        beta = st.slider("Ritmo Beta (Foco)", 0.0, 5.0, 1.0)
        t_eeg = np.linspace(0, 2, 1000)
        eeg = (alfa * np.sin(2*np.pi*10*t_eeg)) + (beta * np.sin(2*np.pi*20*t_eeg)) + np.random.normal(0, 0.5, 1000)
        fig, ax = plt.subplots(2, 1, figsize=(12, 6))
        ax[0].plot(t_eeg, eeg, 'black', lw=0.8); ax[0].set_title("Sinal EEG Bruto (Fronto-Occipital)"); ax[0].set_ylabel("µV")
        freqs = np.fft.rfftfreq(len(t_eeg), 2/1000); power = np.abs(np.fft.rfft(eeg))
        ax[1].plot(freqs[:50], power[:50], color='blue'); ax[1].set_title("Espectro de Frequência (Análise FFT)"); ax[1].set_xlabel("Frequência (Hz)")
        st.pyplot(fig)

    elif modo == "Eletromiograma (EMG / CMAP)":
        st.markdown("**Potencial de Ação Muscular Composto (CMAP)**")
        st.write("Representa a soma espaço-temporal da atividade elétrica das fibras musculares esqueléticas inervadas por um nervo motor após estimulação.")
        
        col1, col2 = st.columns(2)
        with col1:
            recrutamento = st.slider("Fibras Recrutadas pelo Estímulo (%)", 0, 100, 100) / 100.0
        with col2:
            st.info("💡 **Farmacologia Ativa:** Se o TTX estiver ligado na barra lateral, os canais de sódio não abrem e o músculo não responde ao estímulo do nervo (CMAP zero).")

        # Tempo de 0 a 20 ms
        t_cmap = np.linspace(0, 20, 1000)
        cmap_signal = np.zeros_like(t_cmap)
        
        # 1. Artefato do Estímulo (ocorre no momento do choque, ex: t=1.0 ms)
        artefato = 1.0 * np.exp(-((t_cmap - 1.0)**2) / (2 * 0.05**2))
        cmap_signal += artefato
        
        # 2. Resposta Muscular (Onda M Bifásica)
        # Se TTX estiver activo, não há despolarização (onda_m = 0)
        if not ttx:
            latencia = 3.5  # tempo para a condução nervosa + fenda sináptica
            # Simulação de uma onda M bifásica típica (somatório de Gaussinas)
            fase_negativa = 8.0 * recrutamento * np.exp(-((t_cmap - latencia - 1.0)**2) / (2 * 0.8**2))
            fase_positiva = -6.0 * recrutamento * np.exp(-((t_cmap - latencia - 2.5)**2) / (2 * 1.0**2))
            
            # Se TEA estiver activo, a repolarização (fase positiva) é prolongada
            if tea:
                fase_positiva = -4.0 * recrutamento * np.exp(-((t_cmap - latencia - 4.0)**2) / (2 * 2.5**2))
                
            onda_m = fase_negativa + fase_positiva
            cmap_signal += onda_m

        # Ruído elétrico basal do EMG
        cmap_signal += np.random.normal(0, 0.1, len(t_cmap))

        # Plotagem do CMAP
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_cmap, cmap_signal, color='darkgreen', lw=2)
        ax.set_title(f"Eletromiograma Simulado: CMAP (Recrutamento: {recrutamento*100:.0f}%)")
        ax.set_xlabel("Tempo (ms)")
        ax.set_ylabel("Amplitude (mV)")
        
        # Anotações didáticas
        ax.annotate('Artefato de\nEstímulo', xy=(1.0, 1.0), xytext=(1.5, 3.0),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
        
        if not ttx and recrutamento > 0:
            ax.annotate('Latência\n(~3.5ms)', xy=(3.5, 0), xytext=(2.0, -3.0),
                        arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=5))
            ax.annotate('Onda M\n(Despolarização)', xy=(4.5, 8.0 * recrutamento), xytext=(6.0, 6.0 * recrutamento),
                        arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=5))
            
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', lw=0.8, ls='--')
        st.pyplot(fig)
