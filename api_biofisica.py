import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import signal

# --- CONFIGURAÇÃO DO AMBIENTE ---
st.set_page_config(page_title="Simulador de Biofísica UFS", layout="wide")

# --- CONSTANTES FÍSICAS UNIVERSAIS ---
R = 8.314  # Constante dos gases (J/(mol·K))
T = 310    # Temperatura (K) - 37°C
F = 96485  # Constante de Faraday (C/mol)

# --- FUNÇÕES MATEMÁTICAS ---
def calc_nernst(z, ci, co):
    if ci <= 0 or co <= 0: return 0
    return ((R * T) / (z * F)) * np.log(co / ci) * 1000

def calc_ghk(ni, ne, pn, ki, ke, pk, cli, cle, pcl):
    num = (pk * ke) + (pn * ne) + (pcl * cli)
    den = (pk * ki) + (pn * ni) + (pcl * cle)
    if den <= 0: return 0
    return ((R * T) / F) * np.log(num / den) * 1000

# --- BARRA LATERAL: PARÂMETROS E FARMACOLOGIA ---
st.sidebar.title("🧪 Laboratório de Biofísica")

st.sidebar.subheader("💊 Bloqueadores de Canais")
ttx = st.sidebar.toggle("Tetrodotoxina (TTX)", help="Bloqueia canais de Na+. Inverte a resposta ao estímulo (hiperpolarização).")
tea = st.sidebar.toggle("Tetraetilamônio (TEA)", help="Bloqueia canais de K+.")
verapamil = st.sidebar.toggle("Verapamil", help="Bloqueia canais de Ca2+. Elimina o platô cardíaco.")

st.sidebar.divider()

st.sidebar.subheader("🧬 Concentrações e Permeabilidade")
with st.sidebar.expander("Sódio (Na+)", expanded=True):
    na_i = st.slider("[Na+] Interno (mM)", 1.0, 1000.0, 15.0)
    na_e = st.slider("[Na+] Externo (mM)", 1.0, 1000.0, 145.0)
    p_na = st.slider("Permeabilidade P_Na", 0.0, 10.0, 0.02 if not ttx else 0.0)

with st.sidebar.expander("Potássio (K+)", expanded=True):
    k_i = st.slider("[K+] Interno (mM)", 1.0, 1000.0, 150.0)
    k_e = st.slider("[K+] Externo (mM)", 1.0, 1000.0, 5.0)
    p_k = st.slider("Permeabilidade P_K", 0.0, 10.0, 1.0 if not tea else 0.0)

with st.sidebar.expander("Cloreto (Cl-)", expanded=True):
    cl_i = st.slider("[Cl-] Interno (mM)", 1.0, 1000.0, 10.0)
    cl_e = st.slider("[Cl-] Externo (mM)", 1.0, 1000.0, 110.0)
    p_cl = st.slider("Permeabilidade P_Cl", 0.0, 1.0, 0.45)

# Cálculo do Potencial de Repouso Atual
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
    tipo_celula = st.radio("Selecione o Tecido:", ["Neurônio (Hodgkin-Huxley)", "Músculo Esquelético", "Músculo Cardíaco"], horizontal=True)
    st.info(f"O Repouso inicial é de {vm_ghk:.1f} mV.")
    estimulo = st.slider("Intensidade do Estímulo Elétrico", 0.0, 50.0, 20.0)

    if st.button("⚡ Executar Simulação de Potencial de Ação"):
        
        # --- NEURÔNIO (HODGKIN-HUXLEY) ---
        if tipo_celula == "Neurônio (Hodgkin-Huxley)":
            def hh_model(t, y):
                V, m, h, n = y
                g_na = 120.0 if not ttx else 0.0
                g_k = 36.0 if not tea else 0.0
                g_l, e_na, e_k, e_l = 0.3, 50.0, -77.0, -54.4
                
                # Se TTX ativo, estímulo causa hiperpolarização
                estimulo_aplicado = -estimulo if ttx else estimulo
                i_inj = estimulo_aplicado if 5.0 <= t <= 6.0 else 0.0
                
                am = 0.1*(V+40)/(1-np.exp(-(V+40)/10)) if V != -40 else 1.0
                bm = 4.0*np.exp(-(V+65)/18)
                ah = 0.07*np.exp(-(V+65)/20)
                bh = 1.0/(1+np.exp(-(V+35)/10))
                an = 0.01*(V+55)/(1-np.exp(-(V+55)/10)) if V != -55 else 0.1
                bn = 0.125*np.exp(-(V+65)/80)
                
                dvdt = (i_inj - g_na*(m**3)*h*(V-e_na) - g_k*(n**4)*(V-e_k) - g_l*(V-e_l))
                return [dvdt, am*(1-m) - bm*m, ah*(1-h) - bh*h, an*(1-n) - bn*n]

            sol = solve_ivp(hh_model, [0, 50], [-65, 0.05, 0.6, 0.32], t_eval=np.linspace(0, 50, 1000))
            
            # Cálculo de dados para plotagem
            gna = 120 * (sol.y[1]**3) * sol.y[2] if not ttx else np.zeros_like(sol.t)
            gk = 36 * (sol.y[3]**4) if not tea else np.zeros_like(sol.t)
            i_na = gna * (sol.y[0] - 50.0)
            i_k = gk * (sol.y[0] + 77.0)

            # Nova arquitetura com GridSpec para colocar o Plano de Fase (V vs I) ao lado da Condutância
            fig = plt.figure(figsize=(12, 16))
            gs = fig.add_gridspec(4, 2)
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, :])
            ax3 = fig.add_subplot(gs[2, 0])
            ax_vi = fig.add_subplot(gs[2, 1])
            ax4 = fig.add_subplot(gs[3, :])

            ax1.plot(sol.t, sol.y[0], 'purple', lw=2)
            ax1.axvspan(5, 6, color='yellow', alpha=0.3, label='Estímulo Injetado')
            ax1.set_ylabel("Vm (mV)"); ax1.set_title("Potencial de Membrana"); ax1.grid(alpha=0.3); ax1.legend()
            
            ax2.plot(sol.t, i_na, 'g', label="I_Na")
            ax2.plot(sol.t, i_k, 'orange', label="I_K")
            ax2.axhline(0, color='black', lw=1, ls='--')
            ax2.set_ylabel("Corrente (µA/cm²)"); ax2.set_title("Correntes Iônicas (Tempo)"); ax2.grid(alpha=0.3); ax2.legend()
            
            ax3.plot(sol.t, gna, 'g--', label="g_Na")
            ax3.plot(sol.t, gk, 'orange', '--', label="g_K")
            ax3.set_ylabel("Condutância (mS/cm²)"); ax3.set_xlabel("Tempo (ms)"); ax3.set_title("Condutâncias"); ax3.grid(alpha=0.3); ax3.legend()
            
            # Novo Gráfico V x I (Plano de Fase)
            ax_vi.plot(sol.y[0], i_na, 'g', label="I_Na", alpha=0.7)
            ax_vi.plot(sol.y[0], i_k, 'orange', label="I_K", alpha=0.7)
            ax_vi.axhline(0, color='black', lw=1, ls='--')
            ax_vi.axvline(-77, color='orange', lw=1, ls=':', label="E_K (-77mV)")
            ax_vi.axvline(50, color='green', lw=1, ls=':', label="E_Na (+50mV)")
            ax_vi.set_xlabel("Voltagem Vm (mV)"); ax_vi.set_ylabel("Corrente (µA/cm²)")
            ax_vi.set_title("Plano de Fase: V x I (Potencial de Reversão)")
            ax_vi.grid(alpha=0.3); ax_vi.legend(fontsize='small')

            ax4.plot(sol.t, sol.y[1], 'g', label="m (Ativação Na)")
            ax4.plot(sol.t, sol.y[2], 'r', label="h (Inativação Na)")
            ax4.plot(sol.t, sol.y[3], 'orange', label="n (Ativação K)")
            ax4.set_ylabel("Gating (0-1)"); ax4.set_xlabel("Tempo (ms)"); ax4.set_title("Dinâmica de Gating"); ax4.grid(alpha=0.3); ax4.legend()
            
            plt.tight_layout()
            st.pyplot(fig)

        # --- MÚSCULOS ---
        else:
            t_max = 400 if tipo_celula == "Músculo Cardíaco" else 50
            t = np.linspace(0, t_max, 1000); dt = t[1] - t[0]
            v = np.full(1000, vm_ghk)
            gna, gk, gca = np.zeros(1000), np.zeros(1000), np.zeros(1000)
            fase = 0; t_f = 0
            
            # Estímulo negativo (hiperpolarização) se TTX ativo
            estimulo_aplicado = -estimulo if ttx else estimulo

            for i in range(1, 1000):
                if 5.0 <= t[i] <= 7.0: 
                    v[i] = v[i-1] + (estimulo_aplicado * dt * 5)
                else: 
                    v[i] = v[i-1]

                if tipo_celula == "Músculo Esquelético":
                    if v[i] > -55 and fase == 0 and not ttx: fase = 1
                    if fase == 1: 
                        gna[i] = 40; v[i] = min(v[i-1] + 1500*dt, 35)
                        if v[i] >= 34: fase = 2
                    elif fase == 2:
                        if not tea: gk[i] = 20; v[i] = max(v[i-1] - 500*dt, -90)
                        if v[i] <= -89: fase = 3
                    elif fase == 0: 
                        # Retorno passivo. Corrige hiperpolarização do TTX puxando de volta pro Vm_GHK
                        v[i] = v[i-1] - (v[i-1] - vm_ghk) * 0.1
                
                else: # Cardíaco
                    if v[i] > -40 and fase == 0 and not ttx: fase = 1
                    if fase == 1:
                        gna[i] = 30; v[i] = min(v[i-1] + 600*dt, 20)
                        if v[i] >= 19: fase = 4 if verapamil else 2
                    elif fase == 2:
                        v[i] = max(v[i-1] - 150*dt, 5); 
                        if v[i] <= 6: fase, t_f = 3, t[i]
                    elif fase == 3:
                        gca[i] = 10; v[i] = v[i-1] - 0.05*dt
                        if t[i] - t_f > 200: fase = 4
                    elif fase == 4:
                        if not tea: gk[i] = 15; v[i] = max(v[i-1] - 200*dt, -90)
                    elif fase == 0: 
                        v[i] = v[i-1] - (v[i-1] - vm_ghk) * 0.1

            # Correntes simuladas baseadas nas condutâncias e no Vm dinâmico
            i_na = gna * (v - 50.0)
            i_k = gk * (v - (-90.0))
            i_ca = gca * (v - 120.0)

            fig = plt.figure(figsize=(12, 12))
            gs = fig.add_gridspec(3, 2)
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, :])
            ax3 = fig.add_subplot(gs[2, 0])
            ax_vi = fig.add_subplot(gs[2, 1])

            cor = 'red' if "Cardíaco" in tipo_celula else 'green'
            if ttx: cor = 'gray' 
            
            ax1.plot(t, v, color=cor, lw=2.5)
            ax1.axvspan(5, 7, color='yellow', alpha=0.3, label='Estímulo')
            ax1.set_ylabel("Vm (mV)"); ax1.set_title(f"Potencial de Membrana ({tipo_celula})"); ax1.grid(alpha=0.3); ax1.legend()
            
            ax2.plot(t, i_na, 'g', label="I_Na")
            ax2.plot(t, i_k, 'orange', label="I_K")
            if "Cardíaco" in tipo_celula: ax2.plot(t, i_ca, 'purple', label="I_Ca")
            ax2.axhline(0, color='black', lw=1, ls='--')
            ax2.set_ylabel("Corrente"); ax2.set_title("Correntes no Tempo"); ax2.grid(alpha=0.3); ax2.legend()
            
            ax3.plot(t, gna, 'g--', label="g_Na")
            ax3.plot(t, gk, 'orange', '--', label="g_K")
            if "Cardíaco" in tipo_celula: ax3.plot(t, gca, 'purple', '--', label="g_Ca")
            ax3.set_ylabel("Condutância"); ax3.set_xlabel("Tempo (ms)"); ax3.set_title("Condutâncias"); ax3.grid(alpha=0.3); ax3.legend()

            # Novo Gráfico V x I
            ax_vi.plot(v, i_na, 'g', alpha=0.7, label="I_Na")
            ax_vi.plot(v, i_k, 'orange', alpha=0.7, label="I_K")
            if "Cardíaco" in tipo_celula: ax_vi.plot(v, i_ca, 'purple', alpha=0.7, label="I_Ca")
            ax_vi.axhline(0, color='black', lw=1, ls='--')
            ax_vi.axvline(-90, color='orange', lw=1, ls=':', label="E_K (-90mV)")
            ax_vi.axvline(50, color='green', lw=1, ls=':', label="E_Na (+50mV)")
            ax_vi.set_xlabel("Voltagem Vm (mV)"); ax_vi.set_ylabel("Corrente")
            ax_vi.set_title("Plano de Fase: V x I")
            ax_vi.grid(alpha=0.3); ax_vi.legend(fontsize='small')
            
            plt.tight_layout()
            st.pyplot(fig)

# ==========================================================
# ABA 3: SINAIS MACROSCÓPICOS (ECG / EEG / EMG)
# ==========================================================
with tabs[2]:
    st.subheader("Simulação de Bio-sinais Macroscópicos")
    modo = st.radio("Selecione o Exame:", ["Eletrocardiograma (ECG)", "Eletroencefalograma (EEG)", "Eletromiograma (EMG / CMAP)"], horizontal=True)
    st.markdown("---")
    
    if modo == "Eletrocardiograma (ECG)":
        patologia = st.selectbox("Condição Cardíaca:", [
            "Ritmo Sinusal Normal", 
            "Fibrilação Atrial (Patologia Atrial)", 
            "Taquicardia Ventricular (Patologia Ventricular)"
        ])
        
        bpm = 75 if patologia == "Ritmo Sinusal Normal" else (160 if "Taquicardia" in patologia else 90)
        bpm = st.slider("Frequência Cardíaca Média (BPM)", 40, 200, bpm)
        
        fs = 500
        t_ecg = np.linspace(0, 5, 5 * fs)
        ecg = np.zeros_like(t_ecg)
        
        beat_times = []
        curr_t = 0.5
        while curr_t < 5.0:
            beat_times.append(curr_t)
            if patologia == "Fibrilação Atrial (Patologia Atrial)":
                curr_t += np.random.uniform(0.4, 1.2) 
            else:
                curr_t += 60.0 / bpm

        for bt in beat_times:
            tc = t_ecg - bt
            mask = (tc > -0.3) & (tc < 0.6)
            tc_m = tc[mask]
            
            if patologia == "Taquicardia Ventricular (Patologia Ventricular)":
                qrs = 1.5 * np.exp(-((tc_m)**2)/(2*0.06**2))
                t_wave = -0.6 * np.exp(-((tc_m-0.3)**2)/(2*0.05**2))
                ecg[mask] += qrs + t_wave
            else:
                qrs = 1.2 * np.exp(-((tc_m)**2)/(2*0.015**2))
                t_wave = 0.3 * np.exp(-((tc_m-0.35)**2)/(2*0.03**2))
                ecg[mask] += qrs + t_wave
                
                if patologia == "Ritmo Sinusal Normal":
                    p_wave = 0.2 * np.exp(-((tc_m+0.2)**2)/(2*0.015**2))
                    ecg[mask] += p_wave

        if patologia == "Fibrilação Atrial (Patologia Atrial)":
            ecg += 0.05 * np.sin(2 * np.pi * 6 * t_ecg) + 0.03 * np.sin(2 * np.pi * 4.5 * t_ecg + 1)
        
        ecg += np.random.normal(0, 0.02, len(t_ecg))
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t_ecg, ecg, 'r')
        ax.set_title(f"Eletrocardiograma: {patologia}")
        ax.set_xlabel("Tempo (s)"); ax.set_ylabel("mV")
        ax.grid(True, which='both', color='red', alpha=0.2)
        st.pyplot(fig)
    
    elif modo == "Eletroencefalograma (EEG)":
        st.markdown("**Sintetizador de Ondas Cerebrais (Soma de Frequências)**")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: delta = st.slider("Delta (1-4 Hz)", 0.0, 5.0, 1.0)
        with c2: theta = st.slider("Teta (4-8 Hz)", 0.0, 5.0, 0.5)
        with c3: alfa = st.slider("Alfa (8-13 Hz)", 0.0, 5.0, 3.0)
        with c4: beta = st.slider("Beta (13-30 Hz)", 0.0, 5.0, 1.0)
        with c5: gama = st.slider("Gama (30-100 Hz)", 0.0, 5.0, 0.2)
        
        fs_eeg = 250
        t_eeg = np.linspace(0, 4, 4 * fs_eeg)
        
        def gerar_banda(amp, fmin, fmax):
            sinal = np.zeros_like(t_eeg)
            for _ in range(3):
                f = np.random.uniform(fmin, fmax)
                fase = np.random.uniform(0, 2*np.pi)
                sinal += (amp/3) * np.sin(2*np.pi*f*t_eeg + fase)
            return sinal
            
        eeg = gerar_banda(delta, 1, 4) + gerar_banda(theta, 4, 8) + \
              gerar_banda(alfa, 8, 13) + gerar_banda(beta, 13, 30) + \
              gerar_banda(gama, 30, 60) + np.random.normal(0, 0.2, len(t_eeg))
        
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.5])
        
        # 1. Sinal Bruto
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(t_eeg, eeg, color='black', lw=1)
        ax1.set_title("Sinal EEG Bruto (Tempo)")
        ax1.set_ylabel("Amplitude (µV)")
        
        # 2. PSD (Welch)
        ax2 = fig.add_subplot(gs[1])
        nperseg_val = min(256, len(eeg))
        f_welch, Pxx = signal.welch(eeg, fs_eeg, nperseg=nperseg_val)
        ax2.plot(f_welch, Pxx, color='blue')
        ax2.fill_between(f_welch, Pxx, color='blue', alpha=0.3)
        ax2.set_xlim(0, 100) # Integração estendida até 100Hz
        ax2.set_title("Decomposição PSD (Densidade Espectral de Potência)")
        ax2.set_ylabel("Potência")
        
        # 3. Espectrograma de Calor (Fluido)
        ax3 = fig.add_subplot(gs[2])
        noverlap_val = int(nperseg_val * 0.85) # Alta sobreposição para maior fluidez visual
        f_spec, t_spec, Sxx_spec = signal.spectrogram(eeg, fs=fs_eeg, nperseg=nperseg_val, noverlap=noverlap_val)
        Sxx_log = 10 * np.log10(Sxx_spec + 1e-10) # dB
        
        # pcolormesh com shading 'gouraud' cria o efeito de "ilhas de calor" suaves
        im = ax3.pcolormesh(t_spec, f_spec, Sxx_log, shading='gouraud', cmap='turbo')
        ax3.set_ylim(0, 100) # Até 100 Hz
        ax3.set_title("Espectrograma Fluido (Calor Tempo-Frequência)")
        ax3.set_xlabel("Tempo (s)")
        ax3.set_ylabel("Frequência (Hz)")
        fig.colorbar(im, ax=ax3, label="Intensidade (dB)")
        
        plt.tight_layout()
        st.pyplot(fig)

    elif modo == "Eletromiograma (EMG / CMAP)":
        st.markdown("**Potencial de Ação Muscular Composto (CMAP)**")
        recrutamento = st.slider("Fibras Recrutadas pelo Estímulo (%)", 0, 100, 100) / 100.0
        
        t_cmap = np.linspace(0, 20, 1000)
        cmap_signal = np.zeros_like(t_cmap)
        
        artefato = (-1.0 if ttx else 1.0) * np.exp(-((t_cmap - 1.0)**2) / (2 * 0.05**2))
        cmap_signal += artefato
        
        if not ttx:
            latencia = 3.5
            fase_negativa = 8.0 * recrutamento * np.exp(-((t_cmap - latencia - 1.0)**2) / (2 * 0.8**2))
            fase_positiva = -6.0 * recrutamento * np.exp(-((t_cmap - latencia - 2.5)**2) / (2 * 1.0**2))
            if tea:
                fase_positiva = -4.0 * recrutamento * np.exp(-((t_cmap - latencia - 4.0)**2) / (2 * 2.5**2))
            cmap_signal += fase_negativa + fase_positiva

        cmap_signal += np.random.normal(0, 0.1, len(t_cmap))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(t_cmap, cmap_signal, color='darkgreen', lw=2)
        ax.set_title(f"Eletromiograma: CMAP (Recrutamento: {recrutamento*100:.0f}%)")
        ax.set_xlabel("Tempo (ms)"); ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
