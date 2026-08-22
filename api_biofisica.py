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

tabs = st.tabs([
    "📊 Repouso (GHK)", 
    "⚡ Células Excitáveis (PA)", 
    "🫀 Sinais Macroscópicos",
    "👁️ Visão",
    "👂 Audição"
])

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
        fig_repouso, ax_repouso = plt.subplots(figsize=(8, 4))
        ax_repouso.barh(['K+', 'Vm (GHK)', 'Cl-', 'Na+'], 
                        [calc_nernst(1, k_i, k_e), vm_ghk, calc_nernst(-1, cl_i, cl_e), calc_nernst(1, na_i, na_e)],
                        color=['orange', 'blue', 'red', 'green'])
        ax_repouso.axvline(0, color='black', lw=1)
        ax_repouso.set_title("Equilíbrio de Nernst vs Realidade GHK")
        st.pyplot(fig_repouso)

# ==========================================================
# ABA 2: CÉLULAS EXCITÁVEIS (PA DETALHADO)
# ==========================================================
with tabs[1]:
    tipo_celula = st.radio("Selecione o Tecido:", ["Neurônio (Hodgkin-Huxley)", "Músculo Esquelético", "Músculo Cardíaco"], horizontal=True)
    st.info(f"O Repouso inicial é de {vm_ghk:.1f} mV.")
    estimulo = st.slider("Intensidade do Estímulo Elétrico", 0.0, 50.0, 20.0)

    if st.button("⚡ Executar Simulação de Potencial de Ação"):
        
        if tipo_celula == "Neurônio (Hodgkin-Huxley)":
            def hh_model(t, y):
                V, m, h, n = y
                g_na = 120.0 if not ttx else 0.0
                g_k = 36.0 if not tea else 0.0
                g_l, e_na, e_k, e_l = 0.3, 50.0, -77.0, -54.4
                
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
            
            gna = 120 * (sol.y[1]**3) * sol.y[2] if not ttx else np.zeros_like(sol.t)
            gk = 36 * (sol.y[3]**4) if not tea else np.zeros_like(sol.t)
            i_na = gna * (sol.y[0] - 50.0)
            i_k = gk * (sol.y[0] + 77.0)

            fig_hh = plt.figure(figsize=(12, 16))
            gs = fig_hh.add_gridspec(4, 2)
            ax1 = fig_hh.add_subplot(gs[0, :])
            ax2 = fig_hh.add_subplot(gs[1, :])
            ax3 = fig_hh.add_subplot(gs[2, 0])
            ax_vi = fig_hh.add_subplot(gs[2, 1])
            ax4 = fig_hh.add_subplot(gs[3, :])

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
            
            ax_vi.plot(sol.y[0], i_na, 'g', label="I_Na", alpha=0.7)
            ax_vi.plot(sol.y[0], i_k, 'orange', label="I_K", alpha=0.7)
            ax_vi.axhline(0, color='black', lw=1, ls='--')
            ax_vi.axvline(-77, color='orange', lw=1, ls=':', label="E_K (-77mV)")
            ax_vi.axvline(50, color='green', lw=1, ls=':', label="E_Na (+50mV)")
            ax_vi.set_xlabel("Voltagem Vm (mV)"); ax_vi.set_ylabel("Corrente (µA/cm²)")
            ax_vi.set_title("Plano de Fase: V x I")
            ax_vi.grid(alpha=0.3); ax_vi.legend(fontsize='small')

            ax4.plot(sol.t, sol.y[1], 'g', label="m (Ativação Na)")
            ax4.plot(sol.t, sol.y[2], 'r', label="h (Inativação Na)")
            ax4.plot(sol.t, sol.y[3], 'orange', label="n (Ativação K)")
            ax4.set_ylabel("Gating (0-1)"); ax4.set_xlabel("Tempo (ms)"); ax4.set_title("Dinâmica de Gating"); ax4.grid(alpha=0.3); ax4.legend()
            
            plt.tight_layout()
            st.pyplot(fig_hh)

        else:
            t_max = 400 if tipo_celula == "Músculo Cardíaco" else 50
            t = np.linspace(0, t_max, 1000); dt = t[1] - t[0]
            v = np.full(1000, vm_ghk)
            gna, gk, gca = np.zeros(1000), np.zeros(1000), np.zeros(1000)
            fase = 0; t_f = 0
            
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

            i_na = gna * (v - 50.0)
            i_k = gk * (v - (-90.0))
            i_ca = gca * (v - 120.0)

            fig_musc = plt.figure(figsize=(12, 12))
            gs_musc = fig_musc.add_gridspec(3, 2)
            ax1 = fig_musc.add_subplot(gs_musc[0, :])
            ax2 = fig_musc.add_subplot(gs_musc[1, :])
            ax3 = fig_musc.add_subplot(gs_musc[2, 0])
            ax_vi = fig_musc.add_subplot(gs_musc[2, 1])

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
            st.pyplot(fig_musc)

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
        
        fig_ecg, ax_ecg = plt.subplots(figsize=(12, 4))
        ax_ecg.plot(t_ecg, ecg, 'r')
        ax_ecg.set_title(f"Eletrocardiograma: {patologia}")
        ax_ecg.set_xlabel("Tempo (s)"); ax_ecg.set_ylabel("mV")
        ax_ecg.grid(True, which='both', color='red', alpha=0.2)
        st.pyplot(fig_ecg)
    
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
        
        fig_eeg = plt.figure(figsize=(12, 10))
        gs_eeg = fig_eeg.add_gridspec(3, 1, height_ratios=[1, 1, 1.5])
        
        ax1 = fig_eeg.add_subplot(gs_eeg[0])
        ax1.plot(t_eeg, eeg, color='black', lw=1)
        ax1.set_title("Sinal EEG Bruto (Tempo)")
        ax1.set_ylabel("Amplitude (µV)")
        
        ax2 = fig_eeg.add_subplot(gs_eeg[1])
        nperseg_val = min(256, len(eeg))
        f_welch, Pxx = signal.welch(eeg, fs_eeg, nperseg=nperseg_val)
        ax2.plot(f_welch, Pxx, color='blue')
        ax2.fill_between(f_welch, Pxx, color='blue', alpha=0.3)
        ax2.set_xlim(0, 100) 
        ax2.set_title("Decomposição PSD (Densidade Espectral de Potência)")
        ax2.set_ylabel("Potência")
        
        ax3 = fig_eeg.add_subplot(gs_eeg[2])
        noverlap_val = int(nperseg_val * 0.85) 
        f_spec, t_spec, Sxx_spec = signal.spectrogram(eeg, fs=fs_eeg, nperseg=nperseg_val, noverlap=noverlap_val)
        Sxx_log = 10 * np.log10(Sxx_spec + 1e-10) 
        
        im = ax3.pcolormesh(t_spec, f_spec, Sxx_log, shading='gouraud', cmap='turbo')
        ax3.set_ylim(0, 100) 
        ax3.set_title("Espectrograma Fluido (Calor Tempo-Frequência)")
        ax3.set_xlabel("Tempo (s)")
        ax3.set_ylabel("Frequência (Hz)")
        fig_eeg.colorbar(im, ax=ax3, label="Intensidade (dB)")
        
        plt.tight_layout()
        st.pyplot(fig_eeg)

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

        fig_cmap, ax_cmap = plt.subplots(figsize=(10, 5))
        ax_cmap.plot(t_cmap, cmap_signal, color='darkgreen', lw=2)
        ax_cmap.set_title(f"Eletromiograma: CMAP (Recrutamento: {recrutamento*100:.0f}%)")
        ax_cmap.set_xlabel("Tempo (ms)"); ax_cmap.set_ylabel("Amplitude (mV)")
        ax_cmap.grid(True, alpha=0.3)
        st.pyplot(fig_cmap)

# ==========================================================
# ABA 4: BIOFÍSICA DA VISÃO
# ==========================================================
with tabs[3]:
    st.subheader("👁️ Óptica Fisiológica e Fotorrecepção")
    st.markdown("A captação luminosa ocorre na retina, onde os fótons ativam proteínas (opsinas), desencadeando a hiperpolarização da membrana celular através da cascata do GMPc.")
    
    col_vis1, col_vis2 = st.columns([1, 2])
    with col_vis1:
        st.markdown("### Fotorreceptores da Retina")
        st.write("**Bastonetes:** Altamente sensíveis à luz (visão escotópica). Não diferenciam cores. Abundantes na periferia da retina.")
        st.write("**Cones:** Menor sensibilidade à luz, requerem ambientes iluminados (visão fotópica). Responsáveis pela acuidade visual e visão de cores. Concentrados na fóvea.")
        
    with col_vis2:
        st.markdown("### Espectro de Absorção e Visão Comparada")
        especie_visao = st.radio("Selecione o Sistema Visual:", 
                                 ["Humanos (Tricromata)", "Abelhas (UV-Tricromata)", "Águias (Tetracromata)", "Lagartos (Tetracromata + Gotículas)"],
                                 horizontal=True)
        
        def opsina(x, pico, largura):
            return np.exp(-((x - pico)**2) / (2 * largura**2))
        
        ondas = np.linspace(300, 750, 500)
        fig_vis, ax_vis = plt.subplots(figsize=(10, 4))
        
        for wl_bg in range(400, 701, 5):
            cor_bg = plt.cm.turbo((wl_bg - 400) / 300.0) 
            ax_vis.axvspan(wl_bg, wl_bg+5, color=cor_bg, alpha=0.15)
            
        if especie_visao == "Humanos (Tricromata)":
            ax_vis.plot(ondas, opsina(ondas, 420, 20), color='blue', label='Cone S (Azul)', lw=2)
            ax_vis.plot(ondas, opsina(ondas, 534, 25), color='green', label='Cone M (Verde)', lw=2)
            ax_vis.plot(ondas, opsina(ondas, 564, 25), color='red', label='Cone L (Vermelho)', lw=2)
            ax_vis.plot(ondas, opsina(ondas, 498, 30), color='black', label='Bastonetes', ls='--', lw=1.5)
            
        elif especie_visao == "Abelhas (UV-Tricromata)":
            ax_vis.axvspan(300, 400, color='purple', alpha=0.1, label='Zona UV')
            ax_vis.plot(ondas, opsina(ondas, 344, 20), color='purple', label='Receptor UV', lw=2)
            ax_vis.plot(ondas, opsina(ondas, 436, 25), color='blue', label='Receptor Azul', lw=2)
            ax_vis.plot(ondas, opsina(ondas, 544, 30), color='green', label='Receptor Verde', lw=2)
            
        elif especie_visao == "Águias (Tetracromata)":
            ax_vis.plot(ondas, opsina(ondas, 370, 20), color='purple', label='Cone UV/Violeta', lw=2)
            ax_vis.plot(ondas, opsina(ondas, 430, 20), color='blue', label='Cone S', lw=2)
            ax_vis.plot(ondas, opsina(ondas, 500, 20), color='green', label='Cone M', lw=2)
            ax_vis.plot(ondas, opsina(ondas, 560, 20), color='red', label='Cone L', lw=2)
            
        elif especie_visao == "Lagartos (Tetracromata + Gotículas)":
            ax_vis.plot(ondas, opsina(ondas, 360, 12), color='purple', label='Cone UV', lw=2)
            ax_vis.plot(ondas, opsina(ondas, 430, 12), color='blue', label='Cone S (filtrado)', lw=2)
            ax_vis.plot(ondas, opsina(ondas, 495, 12), color='green', label='Cone M (filtrado)', lw=2)
            ax_vis.plot(ondas, opsina(ondas, 570, 12), color='red', label='Cone L (filtrado)', lw=2)

        ax_vis.set_xlim(300, 750)
        ax_vis.set_title("Espectro de Absorção Fotopigmentada")
        ax_vis.set_xlabel("Comprimento de Onda (nm)")
        ax_vis.set_ylabel("Absorção Relativa")
        ax_vis.legend(loc='upper right')
        st.pyplot(fig_vis)

    # RECUO PARA 4 ESPAÇOS: Código flui normalmente dentro da Aba 4
    st.markdown("---")
    st.markdown("### 🎨 Simulador de Ativação de Cones")
    st.write("Deslize para alterar a cor da luz (comprimento de onda) e observe como a retina codifica essa informação através da ativação proporcional dos 3 tipos de cones humanos.")
    
    wl = st.slider("Comprimento de Onda da Luz (nm)", 380, 750, 500)
    
    def wl_to_rgb(wl_val):
        if 380 <= wl_val <= 440: r, g, b = -(wl_val - 440) / (440 - 380), 0.0, 1.0
        elif 440 <= wl_val <= 490: r, g, b = 0.0, (wl_val - 440) / (490 - 440), 1.0
        elif 490 <= wl_val <= 510: r, g, b = 0.0, 1.0, -(wl_val - 510) / (510 - 490)
        elif 510 <= wl_val <= 580: r, g, b = (wl_val - 510) / (580 - 510), 1.0, 0.0
        elif 580 <= wl_val <= 645: r, g, b = 1.0, -(wl_val - 645) / (645 - 580), 0.0
        elif 645 <= wl_val <= 750: r, g, b = 1.0, 0.0, 0.0
        else: r, g, b = 0.0, 0.0, 0.0
        
        fator = 0.3 + 0.7*(wl_val - 380)/(420 - 380) if wl_val < 420 else (0.3 + 0.7*(750 - wl_val)/(750 - 700) if wl_val > 700 else 1.0)
        return int((r*fator)**0.8 * 255), int((g*fator)**0.8 * 255), int((b*fator)**0.8 * 255)

    r_cor, g_cor, b_cor = wl_to_rgb(wl)
    
    ativ_s = opsina(wl, 420, 20)
    ativ_m = opsina(wl, 534, 25)
    ativ_l = opsina(wl, 564, 25)

    col_cor1, col_cor2 = st.columns([1, 2])
    with col_cor1:
        html_color = f"""
        <div style="background-color: rgb({r_cor}, {g_cor}, {b_cor});
                    width: 100%; height: 120px; border-radius: 10px; 
                    border: 2px solid #555; box-shadow: 2px 2px 5px rgba(0,0,0,0.3);">
        </div>
        <p style="text-align:center; font-weight:bold; margin-top:5px;">Luz Incidente: {wl} nm</p>
        """
        st.markdown(html_color, unsafe_allow_html=True)
        
    with col_cor2:
        fig_ativ, ax_ativ = plt.subplots(figsize=(6, 2.5))
        ax_ativ.barh(['Cone L (Vermelho)', 'Cone M (Verde)', 'Cone S (Azul)'], 
                     [ativ_l, ativ_m, ativ_s], 
                     color=['red', 'green', 'blue'])
        ax_ativ.set_xlim(0, 1.1)
        ax_ativ.set_xlabel("Nível de Disparo do Fotorreceptor")
        ax_ativ.set_title("Resposta Neural da Retina Humana")
        ax_ativ.spines['top'].set_visible(False)
        ax_ativ.spines['right'].set_visible(False)
        st.pyplot(fig_ativ)

    st.markdown("---")
    st.markdown("### ⚡ Eletrofisiologia da Visão: A Cascata no Escuro vs Luz")
    st.markdown("Diferente da maioria das células neurais, os **fotorreceptores despolarizam no escuro** (~ -40 mV), liberando o neurotransmissor glutamato constantemente. Quando a luz atinge a opsina, a cascata química fecha canais iônicos, causando **hiperpolarização** (~ -65 mV).")
    
    if st.button("🔦 Simular Incidência de Luz na Retina"):
        t_vis = np.linspace(0, 500, 1000) 
        pulso_luz = np.where((t_vis > 100) & (t_vis < 300), 1.0, 0.0) 
        
        vm_l = -40.0 - (25.0 * ativ_l * pulso_luz)
        vm_m = -40.0 - (25.0 * ativ_m * pulso_luz)
        vm_s = -40.0 - (25.0 * ativ_s * pulso_luz)
        
        ativacao_maxima = max(ativ_l, ativ_m, ativ_s)
        taxa_disparo = 10 + (90 * ativacao_maxima * pulso_luz) 
        
        spikes_nervo = []
        integral = 0
        dt_vis = t_vis[1] - t_vis[0]
        for idx in range(len(t_vis)):
            integral += taxa_disparo[idx] * (dt_vis / 1000.0) 
            if integral >= 1.0:
                spikes_nervo.append(t_vis[idx])
                integral = 0
                
        fig_foto, (ax_luz, ax_vm, ax_nervo) = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2, 1]})
        
        hex_color = f'#{r_cor:02x}{g_cor:02x}{b_cor:02x}'
        ax_luz.plot(t_vis, pulso_luz, color=hex_color, lw=3)
        ax_luz.fill_between(t_vis, pulso_luz, color=hex_color, alpha=0.3)
        ax_luz.set_ylabel("Fótons")
        ax_luz.set_title(f"Estímulo Luminoso (λ = {wl} nm)")
        ax_luz.set_yticks([])
        
        ax_vm.plot(t_vis, vm_l, 'r', label='Cone L (Vermelho)', lw=2)
        ax_vm.plot(t_vis, vm_m, 'g', label='Cone M (Verde)', lw=2)
        ax_vm.plot(t_vis, vm_s, 'b', label='Cone S (Azul)', lw=2)
        ax_vm.axhline(-40, color='gray', ls='--', label='Corrente de Escuro (-40mV)')
        ax_vm.set_ylabel("Vm (mV)")
        ax_vm.set_title("Potencial de Membrana do Fotorreceptor (Hiperpolarização)")
        ax_vm.legend()
        ax_vm.grid(alpha=0.3)
        
        ax_nervo.vlines(spikes_nervo, ymin=0, ymax=1, color='purple')
        ax_nervo.set_ylabel("Spikes")
        ax_nervo.set_xlabel("Tempo (ms)")
        ax_nervo.set_title("Via Direta: Ativação da Célula Ganglionar ON e Nervo Óptico")
        ax_nervo.set_yticks([])
        
        plt.tight_layout()
        st.pyplot(fig_foto)

# ==========================================================
# ABA 5: BIOFÍSICA DA AUDIÇÃO
# ==========================================================
with tabs[4]:
    st.subheader("👂 Acústica Fisiológica e Mecanotransdução")
    st.markdown("O som é uma onda mecânica convertida em sinal elétrico na cóclea pelo movimento dos estereocílios das células ciliadas.")
    
    col_aud1, col_aud2 = st.columns([1, 2])
    with col_aud1:
        st.markdown("### As Células Ciliadas")
        st.write("**Internas (CCI):** Verdadeiros receptores sensoriais (transdutores). O movimento mecânico abre canais de K+ dependentes de estiramento.")
        st.write("**Externas (CCE):** Atuam como amplificadores cocleares. Apresentam *eletromotilidade* (encolhem e esticam via proteína prestina) para amplificar sons fracos.")
        
        st.markdown("---")
        st.markdown("### Espectro Sonoro Animal")
        animais = ['Humano', 'Cão', 'Ave (Pombo)', 'Morcego']
        min_hz = [20, 67, 100, 10000]
        max_hz = [20000, 45000, 8000, 200000]
        
        fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
        for idx_anim in range(len(animais)):
            ax_bar.barh(animais[idx_anim], max_hz[idx_anim] - min_hz[idx_anim], left=min_hz[idx_anim], color='teal', alpha=0.7)
        ax_bar.set_xscale('log')
        ax_bar.set_xlabel("Frequência (Hz) - Escala Log")
        ax_bar.set_title("Capacidade Auditiva")
        st.pyplot(fig_bar)

    with col_aud2:
        st.markdown("### Tonotopia e Presbiacusia (Envelhecimento)")
        freqs_audiograma = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        
        idade_bebe = [0, -2, 0, 0, 0, 0, 5, 5]
        idade_jovem = [5, 5, 5, 5, 5, 10, 15, 25]
        idade_idoso = [15, 15, 20, 25, 40, 60, 80, 110]
        
        fig_aud, ax_aud = plt.subplots(figsize=(10, 5))
        ax_aud.plot(freqs_audiograma, idade_bebe, 'o-', color='blue', label='Bebê (Cóclea Intacta)', lw=2)
        ax_aud.plot(freqs_audiograma, idade_jovem, 's-', color='green', label='Jovem Adulto', lw=2)
        ax_aud.plot(freqs_audiograma, idade_idoso, '^-', color='red', label='Idoso (Presbiacusia)', lw=2)
        
        ax_aud.set_xscale('log')
        ax_aud.set_xticks(freqs_audiograma)
        ax_aud.set_xticklabels(['125', '250', '500', '1k', '2k', '4k', '8k', '16k'])
        ax_aud.set_ylim(-10, 120)
        ax_aud.invert_yaxis()
        
        ax_aud.axhspan(-10, 20, color='gray', alpha=0.1, label='Audição Normal')
        ax_aud.set_title("Audiograma Clínico Simulado")
        ax_aud.set_xlabel("Frequência do Som (Hz) - Do Grave ao Agudo")
        ax_aud.set_ylabel("Limiar Auditivo (dB HL) - Escala Invertida")
        ax_aud.legend(loc='lower left')
        ax_aud.grid(True, which='both', ls='--', alpha=0.5)
        st.pyplot(fig_aud)

    # RECUO PARA 4 ESPAÇOS: Código flui normalmente dentro da Aba 5
    st.markdown("---")
    st.markdown("### 🎹 Simulador de Tonotopia Coclear e Ativação Ciliada")
    st.write("Altere a Frequência (Pitch) e a Amplitude (Volume). A frequência determina **onde** a membrana vibra (Tonotopia). A amplitude determina **a força** com que os estereocílios se deformam.")

    col_som1, col_som2 = st.columns(2)
    with col_som1:
        freq_som = st.slider("Frequência da Onda Sonora (Hz)", 20, 20000, 1000, step=10, format="%d Hz")
    with col_som2:
        amp_som = st.slider("Amplitude Sonora (Volume em dB)", 0, 120, 60, help="0 dB = Limiar da Audição | 120 dB = Limiar da Dor")
        amp_linear = amp_som / 120.0 
    
    posicao_ativada = 35 * (1 - np.log10(freq_som / 20) / np.log10(20000 / 20))
    pos = np.linspace(0, 35, 500)
    envelope = amp_linear * np.exp(-((pos - posicao_ativada)**2) / (2 * 1.5**2))
    
    fig_tono, ax_tono = plt.subplots(figsize=(10, 3))
    ax_tono.fill_between(pos, envelope, color='dodgerblue', alpha=0.4)
    ax_tono.plot(pos, envelope, color='navy', lw=2)
    ax_tono.axvline(posicao_ativada, color='red', ls='--', lw=2, label=f'Pico de Ressonância ({posicao_ativada:.1f} mm)')
    
    ax_tono.axvline(0, color='black', lw=1)
    ax_tono.text(0.5, 1.1 * max(1, amp_linear), "BASE\nSons Agudos", ha='left', fontsize=10, fontweight='bold')
    ax_tono.axvline(35, color='black', lw=1)
    ax_tono.text(34.5, 1.1 * max(1, amp_linear), "ÁPICE\nSons Graves", ha='right', fontsize=10, fontweight='bold')

    ax_tono.set_xlim(-2, 37)
    ax_tono.set_ylim(0, 1.4)
    ax_tono.set_xlabel("Distância ao longo da Membrana Basilar da Cóclea (mm)")
    ax_tono.set_yticks([]) 
    ax_tono.legend(loc='upper right')
    
    ax_tono.spines['top'].set_visible(False)
    ax_tono.spines['right'].set_visible(False)
    ax_tono.spines['left'].set_visible(False)
    st.pyplot(fig_tono)

    st.markdown("---")
    st.markdown("### ⚡ Eletrofisiologia da Audição (Transdução Mecanoelétrica)")
    
    if st.button("🔊 Simular Resposta da Célula Ciliada"):
        t_aud = np.linspace(0, 100, 1000) 
        som = np.where((t_aud > 20) & (t_aud < 80), amp_linear, 0.0) 
        
        vm_ciliada = -70.0 + (30.0 * som)
        contracao_cce = 10.0 * som 
        
        taxa_disparo_aud = 5 + (150 * som) 
        spikes_aud = []
        integral_aud = 0
        dt_aud = t_aud[1] - t_aud[0]
        
        for idx in range(len(t_aud)):
            integral_aud += taxa_disparo_aud[idx] * (dt_aud / 1000.0)
            if integral_aud >= 1.0:
                spikes_aud.append(t_aud[idx])
                integral_aud = 0
                
        fig_mec, (ax_som, ax_vm_aud, ax_nervo_aud) = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 2, 1]})
        
        ax_som.plot(t_aud, som, color='gray', lw=2)
        ax_som.fill_between(t_aud, som, color='gray', alpha=0.3)
        ax_som.set_ylabel("Pressão (Pa)")
        ax_som.set_title(f"Estímulo Acústico Recebido ({freq_som} Hz | {amp_som} dB)")
        ax_som.set_ylim(0, 1.1)
        
        ax_vm_aud.plot(t_aud, vm_ciliada, 'teal', lw=2, label='Cél. Ciliada Interna (Vm)')
        
        ax_mec2 = ax_vm_aud.twinx()
        ax_mec2.plot(t_aud, contracao_cce, 'magenta', lw=2, ls='--', label='Contração CCE (%)')
        
        ax_vm_aud.axhline(-70, color='gray', ls=':', label='Repouso (-70mV)')
        ax_vm_aud.set_ylabel("Vm (mV)", color='teal')
        ax_mec2.set_ylabel("Encurtamento Muscular (%)", color='magenta')
        ax_vm_aud.set_title("Despolarização e Eletromotilidade (Na Posição de Pico)")
        
        lines_1, labels_1 = ax_vm_aud.get_legend_handles_labels()
        lines_2, labels_2 = ax_mec2.get_legend_handles_labels()
        ax_vm_aud.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        
        ax_nervo_aud.vlines(spikes_aud, ymin=0, ymax=1, color='navy')
        ax_nervo_aud.set_ylabel("Spikes")
        ax_nervo_aud.set_xlabel("Tempo (ms)")
        ax_nervo_aud.set_title("Fibras do Nervo Coclear")
        ax_nervo_aud.set_yticks([])
        
        plt.tight_layout()
        st.pyplot(fig_mec)
