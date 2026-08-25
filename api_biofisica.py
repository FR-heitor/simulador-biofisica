import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy import signal
import matplotlib.patches as patches

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

def length_to_rgb(wavelength):
    """Converte comprimento de onda (nm) para cor RGB visível."""
    gamma = 0.8
    intensity_max = 255
    factor = 0.0
    R_val, G_val, B_val = 0, 0, 0

    if 380 <= wavelength <= 440:
        R_val = -(wavelength - 440) / (440 - 380)
        G_val = 0.0
        B_val = 1.0
    elif 440 < wavelength <= 490:
        R_val = 0.0
        G_val = (wavelength - 440) / (490 - 440)
        B_val = 1.0
    elif 490 < wavelength <= 510:
        R_val = 0.0
        G_val = 1.0
        B_val = -(wavelength - 510) / (510 - 490)
    elif 510 < wavelength <= 580:
        R_val = (wavelength - 510) / (580 - 510)
        G_val = 1.0
        B_val = 0.0
    elif 580 < wavelength <= 645:
        R_val = 1.0
        G_val = -(wavelength - 645) / (645 - 580)
        B_val = 0.0
    elif 645 < wavelength <= 750:
        R_val = 1.0
        G_val = 0.0
        B_val = 0.0

    if 380 <= wavelength <= 420: factor = 0.3 + 0.7 * (wavelength - 380) / (420 - 380)
    elif 420 < wavelength <= 700: factor = 1.0
    elif 700 < wavelength <= 750: factor = 0.3 + 0.7 * (750 - wavelength) / (750 - 700)

    R_rgb = int(intensity_max * ((R_val * factor) ** gamma)) if R_val > 0 else 0
    G_rgb = int(intensity_max * ((G_val * factor) ** gamma)) if G_val > 0 else 0
    B_rgb = int(intensity_max * ((B_val * factor) ** gamma)) if B_val > 0 else 0

    return f'#{R_rgb:02x}{G_rgb:02x}{B_rgb:02x}'

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

vm_ghk = calc_ghk(na_i, na_e, p_na, k_i, k_e, p_k, cl_i, cl_e, p_cl)

# --- ECRÃ PRINCIPAL ---
st.title("🔬 Plataforma de Estudos Biofísicos")

tabs = st.tabs([
    "📊 Repouso (GHK)", 
    "🔌 Propriedades Passivas (RC)", 
    "⚡ Células Excitáveis (PA)", 
    "🫀 Sinais Macroscópicos", 
    "👁️ Visão", 
    "👂 Audição",
    "🧠 Epilepsia"
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
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(['K+', 'Vm (GHK)', 'Cl-', 'Na+'], 
                [calc_nernst(1, k_i, k_e), vm_ghk, calc_nernst(-1, cl_i, cl_e), calc_nernst(1, na_i, na_e)],
                color=['orange', 'blue', 'red', 'green'])
        ax.axvline(0, color='black', lw=1)
        ax.set_title("Equilíbrio de Nernst vs Realidade GHK")
        st.pyplot(fig)

# ==========================================================
# ABA 2: CIRCUITO ELÉTRICO PASSIVO DA MEMBRANA
# ==========================================================
with tabs[1]:
    st.header("🔌 Comportamento Elétrico Passivo (Circuito RC)")
    
    st.markdown("""
    De acordo com o **Modelo do Mosaico Fluido** (Singer & Nicolson, 1972), a matriz lipídica atua como um isolante, 
    separando as cargas dos meios intra e extracelular (funcionando como um **Capacitor, C**). As proteínas intrínsecas 
    funcionam como vias condutoras de íons, conferindo resistência elétrica ao fluxo (funcionando como um **Resistor, R**).
    
    A resposta da célula a um pulso elétrico obedece à dinâmica de um circuito RC em paralelo, sendo a curva governada 
    pela **Constante de Tempo da Membrana ($\tau = R \cdot C$)**.
    """)
    
    col_rc1, col_rc2 = st.columns([1, 2])
    with col_rc1:
        st.subheader("Parâmetros do Circuito")
        R_m = st.slider("Resistência da Membrana (Rm) [kΩ·cm²]", 0.5, 10.0, 2.0, step=0.5)
        C_m = st.slider("Capacitância da Membrana (Cm) [µF/cm²]", 0.5, 3.0, 1.0, step=0.1)
        
        tau = R_m * C_m # Constante de tempo em milissegundos
        st.success(f"**Constante de Tempo ($\tau$):** {tau:.2f} ms")
        st.info(f"O tempo $\\tau$ ({tau:.2f} ms) é o momento em que a membrana atinge **63%** da sua carga máxima, ou cai para **37%** durante a descarga.")

    with col_rc2:
        t_rc = np.linspace(0, max(20, tau * 5), 500)
        t_inj_start = 2.0
        t_inj_end = t_inj_start + (tau * 3) 
        
        V_max = 100.0 
        V_rc = np.zeros_like(t_rc)
        
        for i, t in enumerate(t_rc):
            if t < t_inj_start:
                V_rc[i] = 0
            elif t_inj_start <= t <= t_inj_end:
                t_active = t - t_inj_start
                V_rc[i] = V_max * (1 - np.exp(-t_active / tau))
            else:
                t_decay = t - t_inj_end
                V_peak = V_max * (1 - np.exp(-(t_inj_end - t_inj_start) / tau))
                V_rc[i] = V_peak * np.exp(-t_decay / tau)

        fig_rc, ax_rc = plt.subplots(figsize=(8, 4))
        ax_rc.plot(t_rc, V_rc, 'b-', lw=2.5, label='Potencial da Membrana V(t)')
        
        stim_square = np.where((t_rc >= t_inj_start) & (t_rc <= t_inj_end), V_max, 0)
        ax_rc.plot(t_rc, stim_square, 'r--', alpha=0.5, label='Pulso de Corrente (Ideal)')
        
        t_tau_charge = t_inj_start + tau
        v_tau_charge = V_max * 0.63
        ax_rc.plot(t_tau_charge, v_tau_charge, 'go', markersize=8, label=f'$\\tau$ (Carga 63%)')
        ax_rc.vlines(t_tau_charge, 0, v_tau_charge, color='g', linestyles='dotted')
        
        ax_rc.set_title("Resposta Exponencial Passiva da Membrana (Carga e Descarga)")
        ax_rc.set_xlabel("Tempo (ms)")
        ax_rc.set_ylabel("Voltagem Relativa (%)")
        ax_rc.grid(True, alpha=0.3)
        ax_rc.legend()
        st.pyplot(fig_rc)

# ==========================================================
# ABA 3: CÉLULAS EXCITÁVEIS (PA DETALHADO)
# ==========================================================
with tabs[2]:
    tipo_celula = st.radio("Selecione o Tecido:", ["Neurônio (Hodgkin-Huxley)", "Músculo Esquelético", "Músculo Cardíaco"], horizontal=True)
    st.info(f"O Repouso inicial é de {vm_ghk:.1f} mV. Lembre-se: O TTX inverte o estímulo causando hiperpolarização.")
    estimulo = st.slider("Intensidade do Estímulo Elétrico", 0.0, 50.0, 20.0)

    if st.button("⚡ Executar Simulação de Potencial de Ação"):
        if tipo_celula == "Neurônio (Hodgkin-Huxley)":
            def hh_model(t, y):
                V, m, h, n = y
                g_na = 120.0 if not ttx else 0.0
                g_k = 36.0 if not tea else 0.0
                g_l, e_na, e_k, e_l = 0.3, 50.0, -77.0, -54.4
                
                # Se TTX ativo, estímulo causa hiperpolarização profunda como proteção
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

        else:
            t_max = 400 if tipo_celula == "Músculo Cardíaco" else 50
            t = np.linspace(0, t_max, 1000); dt = t[1] - t[0]
            v = np.full(1000, vm_ghk)
            gna, gk, gca = np.zeros(1000), np.zeros(1000), np.zeros(1000)
            fase = 0; t_f = 0
            
            # TTX inverte o estímulo (gera hiperpolarização)
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
                
                else:
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
# ABA 4: SINAIS MACROSCÓPICOS (ECG / EEG / EMG)
# ==========================================================
with tabs[3]:
    st.subheader("Simulação de Bio-sinais Macroscópicos")
    modo = st.radio("Selecione o Exame:", ["Eletrocardiograma (ECG)", "Eletroencefalograma (EEG)", "Eletromiograma (EMG / CMAP)"], horizontal=True)
    st.markdown("---")
    
    if modo == "Eletrocardiograma (ECG)":
        patologia = st.selectbox("Condição Cardíaca:", ["Ritmo Sinusal Normal", "Fibrilação Atrial (Patologia Atrial)", "Taquicardia Ventricular (Patologia Ventricular)"])
        bpm = 75 if patologia == "Ritmo Sinusal Normal" else (160 if "Taquicardia" in patologia else 90)
        bpm = st.slider("Frequência Cardíaca Média (BPM)", 40, 200, bpm)
        fs = 500
        t_ecg = np.linspace(0, 5, 5 * fs)
        ecg = np.zeros_like(t_ecg)
        beat_times = []
        curr_t = 0.5
        while curr_t < 5.0:
            beat_times.append(curr_t)
            if patologia == "Fibrilação Atrial (Patologia Atrial)": curr_t += np.random.uniform(0.4, 1.2) 
            else: curr_t += 60.0 / bpm

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
        st.markdown("""
        ### Neurociência Sistêmica e Codificação Preditiva
        O cérebro atua ativamente para **cancelar estímulos previsíveis** e economizar energia metabólica. 
        Abaixo, você pode aplicar estímulos repetitivos. Observe como o interneurônio inibitório **aprende** o ritmo 
        e começa a disparar antecipadamente, suprimindo o neurônio excitatório cortical e o sinal do EEG.
        """)
        
        c1, c2, c3 = st.columns(3)
        with c1: 
            tipo_est = st.radio("Modalidade Sensorial:", ["Calor", "Pressão", "Dor"])
        with c2:
            if tipo_est == "Calor": 
                intensidade = st.slider("Temperatura (°C)", 36.0, 45.0, 39.0)
                int_norm = (intensidade - 36) / 9.0
            elif tipo_est == "Pressão": 
                intensidade = st.slider("Pressão (kPa)", 0, 100, 50)
                int_norm = intensidade / 100.0
            else: 
                intensidade = st.select_slider("Escala Visual de Dor (EVA)", options=["😃", "🙂", "😐", "😟", "😫", "😭"], value="😟")
                mapping = {"😃":0.0, "🙂":0.2, "😐":0.4, "😟":0.6, "😫":0.8, "😭":1.0}
                int_norm = mapping[intensidade]
        
        fs_eeg = 250
        t_eeg = np.linspace(0, 4, 4 * fs_eeg)
        eeg_bg = np.random.normal(0, 0.1, len(t_eeg)) + 0.5*np.sin(2*np.pi*10*t_eeg)
        
        t_sim = np.linspace(0, 4000, 4 * fs_eeg)
        n_aferente = np.zeros_like(t_sim) - 70
        n_inibitorio = np.zeros_like(t_sim) - 70
        n_cortical = np.zeros_like(t_sim) - 70
        erp_sinal = np.zeros_like(t_eeg)
        
        stim_times = [500, 1500, 2500, 3500] 
        aprendizado = [0.0, 0.3, 0.7, 0.95] 
        
        for idx, s_time in enumerate(stim_times):
            pico = int(s_time / (4000 / len(t_sim)))
            
            # Aleatorização do estímulo (variação natural de intensidade/ruído da via aferente)
            fator_ruido = np.random.uniform(0.85, 1.15)
            estimulo_real = int_norm * fator_ruido
            
            if pico < len(n_aferente): 
                n_aferente[pico:pico+10] = 30 * estimulo_real
            
            # Antecipação temporal: o interneurônio dispara ms ANTES do pico após aprender o ritmo
            forca_inibitoria = int_norm * aprendizado[idx]
            ante_pico = pico - int(60 * aprendizado[idx]) 
            if ante_pico > 0 and ante_pico < len(n_inibitorio):
                n_inibitorio[ante_pico:ante_pico+15] = 30 * forca_inibitoria
            
            # Erro de Predição (Sinal Cortical) = Realidade - Inibição (Predição)
            forga_cortical = max(0.0, estimulo_real - forca_inibitoria)
            if pico < len(n_cortical):
                n_cortical[pico:pico+15] = 30 * forga_cortical
                
            # O ERP no EEG macroscópico reflete diretamente este erro de predição cortical
            erp_shape = 5.0 * forga_cortical * np.exp(-((t_eeg - (s_time/1000))**2)/(2*0.05**2))
            erp_sinal += erp_shape
            
        eeg_final = eeg_bg + erp_sinal
        
        for arr in [n_aferente, n_inibitorio, n_cortical]:
            for i in range(1, len(arr)):
                if arr[i] == -70 and arr[i-1] > -70: arr[i] = arr[i-1] - 5
                elif arr[i] < -70: arr[i] = -70

        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.5])
        
        ax_micro = fig.add_subplot(gs[0])
        ax_micro.plot(t_sim, n_aferente + 100, 'g', label='Neurônio Aferente (Realidade)')
        ax_micro.plot(t_sim, n_inibitorio, 'b', label='Interneurônio Inibitório (Predição)')
        ax_micro.plot(t_sim, n_cortical - 100, 'purple', label='Cortical Excitatório (Erro de Predição)')
        for st_time in stim_times: ax_micro.axvline(st_time, color='r', ls='--', alpha=0.5)
        ax_micro.set_title("Micro-escala: Rede Neural de Codificação Preditiva")
        ax_micro.set_yticks([]); ax_micro.legend(loc='upper right')
        
        ax_macro = fig.add_subplot(gs[1])
        ax_macro.plot(t_eeg, eeg_final, color='black', lw=1.2)
        ax_macro.set_title("Macro-escala: Eletroencefalograma (Habituação do ERP)")
        ax_macro.set_xlabel("Tempo (s)"); ax_macro.set_ylabel("Amplitude (µV)")
        for st_time in stim_times: ax_macro.axvline(st_time/1000, color='r', ls='--', alpha=0.5)
        
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

# ==========================================================
# ABA 5: VISÃO
# ==========================================================
with tabs[4]:
    st.header("👁️ Óptica Fisiológica e Fototransdução")
    
    st.markdown("""
    A retina humana possui dois tipos principais de fotorreceptores: os **bastonetes** (altamente sensíveis, visão noturna/escotópica, sem cores) 
    e os **cones** (menor sensibilidade, visão diurna/fotópica, responsáveis pela percepção de cores).
    """)
    
    col_vis1, col_vis2 = st.columns([1, 2])
    with col_vis1:
        especie = st.selectbox("Evolução do Espectro Visual:", [
            "Humano (Tricromata)", 
            "Cão/Gato (Dicromata)", 
            "Abelha (Tricromata + UV)", 
            "Águia (Tetracromata)", 
            "Lagarto Diurno (Tetracromata Filtrado)"
        ])
        
        wl_user = st.slider("Dispare um Fóton (Comprimento de Onda em nm):", 300, 750, 500)
        st.markdown(f"**Cor Estimada da Luz:**")
        st.markdown(f"<div style='width: 100%; height: 50px; background-color: {length_to_rgb(wl_user)}; border-radius: 5px; border: 1px solid #ccc;'></div>", unsafe_allow_html=True)
    
    with col_vis2:
        wl = np.linspace(300, 750, 500)
        curvas = {}
        
        if especie == "Humano (Tricromata)":
            curvas["Bastonete (Rodopsina)"] = (500, 40, 'gray')
            curvas["Cone S (Azul)"] = (420, 35, 'blue')
            curvas["Cone M (Verde)"] = (530, 45, 'green')
            curvas["Cone L (Vermelho)"] = (560, 50, 'red')
        elif especie == "Cão/Gato (Dicromata)":
            curvas["Bastonete"] = (500, 40, 'gray')
            curvas["Cone S (Azul/UV)"] = (430, 40, 'blue')
            curvas["Cone M/L (Amarelo)"] = (555, 50, 'orange')
        elif especie == "Abelha (Tricromata + UV)":
            curvas["Receptor UV"] = (340, 30, 'purple')
            curvas["Receptor Azul"] = (430, 35, 'blue')
            curvas["Receptor Verde"] = (540, 40, 'green')
        elif especie == "Águia (Tetracromata)":
            curvas["Cone UV/Violeta"] = (400, 30, 'purple')
            curvas["Cone S (Azul)"] = (450, 30, 'blue')
            curvas["Cone M (Verde)"] = (530, 35, 'green')
            curvas["Cone L (Vermelho)"] = (570, 35, 'red')
        elif especie == "Lagarto Diurno (Tetracromata Filtrado)":
            curvas["Cone UV"] = (360, 12, 'purple')
            curvas["Cone S"] = (460, 12, 'blue')
            curvas["Cone M"] = (540, 12, 'green')
            curvas["Cone L"] = (610, 12, 'red')

        fig_vis, ax_vis = plt.subplots(figsize=(10, 5))
        ativacoes = {}
        
        for nome, (pico, largura, cor) in curvas.items():
            absorcao = np.exp(-((wl - pico)**2) / (2 * largura**2))
            ax_vis.plot(wl, absorcao, color=cor, label=nome, lw=2)
            ax_vis.fill_between(wl, absorcao, color=cor, alpha=0.1)
            ativacoes[nome] = np.exp(-((wl_user - pico)**2) / (2 * largura**2))
            
        ax_vis.axvline(wl_user, color='black', linestyle='--', label=f'Fóton ({wl_user} nm)')
        ax_vis.set_title(f"Espectro de Absorção das Opsinas: {especie}")
        ax_vis.set_xlabel("Comprimento de Onda (nm)")
        ax_vis.set_ylabel("Absorção Relativa (%)")
        ax_vis.legend(loc='upper right', fontsize='small')
        ax_vis.grid(True, alpha=0.3)
        st.pyplot(fig_vis)

    st.markdown("---")
    st.subheader("Cascata de Fototransdução Neural")
    
    c_foto1, c_foto2 = st.columns([1, 2])
    with c_foto1:
        st.write("**Nível de Ativação do Receptor:**")
        for nome, val in ativacoes.items():
            st.progress(float(val), text=f"{nome}: {val*100:.1f}%")
            
    with c_foto2:
        t_foto = np.linspace(0, 100, 500)
        fig_vm, ax_vm = plt.subplots(figsize=(10, 4))
        
        vm_base = -40 
        for nome, val in ativacoes.items():
            if "Bastonete" in nome or "Cone" in nome or "Receptor" in nome:
                drop = 25 * val 
                vm_curve = vm_base - drop * np.exp(-((t_foto - 20)**2)/200) * (t_foto > 10)
                cor_plot = 'gray' if 'Bastonete' in nome else ('blue' if 'S' in nome or 'Azul' in nome else ('green' if 'M' in nome or 'Verde' in nome else ('red' if 'L' in nome or 'Vermelho' in nome else 'purple')))
                ax_vm.plot(t_foto, vm_curve, color=cor_plot, lw=2, label=nome)
                
        ax_vm.set_title("Potencial de Membrana Fotorreceptor (A Luz causa Hiperpolarização)")
        ax_vm.set_xlabel("Tempo (ms)")
        ax_vm.set_ylabel("Vm (mV)")
        ax_vm.axvspan(10, 30, color='yellow', alpha=0.2, label='Flash de Luz')
        ax_vm.grid(alpha=0.3); ax_vm.legend()
        st.pyplot(fig_vm)

# ==========================================================
# ABA 6: AUDIÇÃO
# ==========================================================
with tabs[5]:
    st.header("👂 Acústica Fisiológica e Mecanotransdução")
    
    st.markdown("""
    A cóclea é organizada de forma **tonotópica**: a base é rígida e responde a altas frequências (sons agudos), 
    enquanto o ápice é flexível e responde a baixas frequências (sons graves).
    """)
    
    col_aud1, col_aud2 = st.columns([1, 2])
    with col_aud1:
        st.subheader("Audiograma Clínico")
        perfil_aud = st.radio("Perfil do Paciente:", ["Jovem Saudável", "Idoso (Presbiacusia)", "Cão", "Morcego (Ultrassom)"])
        freq_user = st.slider("Frequência da Onda Sonora (Hz):", 20, 20000, 2000)
        amp_user = st.slider("Volume do Som (Amplitude dB):", 0, 120, 60)
        
    with col_aud2:
        freqs_audiograma = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
        perdas = {"Jovem Saudável": [0, 0, 5, 0, 5, 5, 10, 15],
                  "Idoso (Presbiacusia)": [10, 15, 20, 25, 40, 60, 80, 95],
                  "Cão": [-10, -5, 0, 0, -10, -15, -20, -10],
                  "Morcego (Ultrassom)": [80, 70, 50, 30, 10, 0, -10, -20]}
        
        fig_aud, ax_aud = plt.subplots(figsize=(8, 4))
        ax_aud.plot(freqs_audiograma, perdas[perfil_aud], 'o-', color='red' if 'Idoso' in perfil_aud else 'blue', lw=2)
        ax_aud.set_xscale('log')
        ax_aud.set_xticks(freqs_audiograma)
        ax_aud.set_xticklabels(freqs_audiograma)
        ax_aud.invert_yaxis()
        ax_aud.set_title("Audiograma Clínico Tonal")
        ax_aud.set_xlabel("Frequência (Hz)")
        ax_aud.set_ylabel("Limiar Auditivo (dB HL)")
        ax_aud.grid(True, which="both", ls="-", alpha=0.3)
        st.pyplot(fig_aud)

    st.markdown("---")
    
    st.subheader("Potenciais Microfônicos (Experimento de Tasaki, 1954)")
    f_micro = st.radio("Selecione a frequência do som (kHz):", [0.5, 1.0, 2.0, 4.0], horizontal=True)
    t_mic = np.linspace(0, 4, 400) 
    
    base_amp = 1.0
    if f_micro == 0.5: apex_amp = 1.0
    elif f_micro == 1.0: apex_amp = 0.6
    elif f_micro == 2.0: apex_amp = 0.15
    elif f_micro == 4.0: apex_amp = 0.0

    onda_base = base_amp * np.sin(2 * np.pi * f_micro * t_mic)
    onda_apex = apex_amp * np.sin(2 * np.pi * f_micro * t_mic)
    onda_som = 1.0 * np.sin(2 * np.pi * f_micro * t_mic)

    fig_mic, axes_mic = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
    
    axes_mic[0].plot(t_mic, onda_som, color='black', lw=1.5)
    axes_mic[0].set_title(f"Som ({f_micro} kHz)")
    axes_mic[0].axis('off')
    
    axes_mic[1].plot(t_mic, onda_base, color='blue', lw=1.5)
    axes_mic[1].set_title("Microfônicos (Giro Basal)")
    axes_mic[1].axhline(0, color='black', lw=0.5, ls='--')
    axes_mic[1].axis('off')
    
    if apex_amp > 0: axes_mic[2].plot(t_mic, onda_apex, color='red', lw=1.5)
    else: axes_mic[2].axhline(0, color='red', lw=1.5) 
    axes_mic[2].set_title("Microfônicos (3º Giro - Ápice)")
    axes_mic[2].axhline(0, color='black', lw=0.5, ls='--')
    axes_mic[2].axis('off')
    st.pyplot(fig_mic)
    
    st.markdown("---")

    c_aud1, c_aud2 = st.columns([1, 2])
    with c_aud1:
        st.write("Eletromotilidade da CCE")
        contração = (amp_user / 120.0) * 20 
        fig_cel, ax_cel = plt.subplots(figsize=(4, 6))
        ax_cel.add_patch(patches.Rectangle((0.2, 0), 0.2, 1.0, facecolor='lightblue', edgecolor='black'))
        ax_cel.add_patch(patches.Rectangle((0.6, 0), 0.2, 1.0 - (contração/100), facecolor='magenta', edgecolor='black'))
        ax_cel.text(0.3, 0.5, 'CCI', ha='center', va='center')
        ax_cel.text(0.7, 0.5, 'CCE', ha='center', va='center')
        ax_cel.set_ylim(0, 1.2)
        ax_cel.axis('off')
        st.pyplot(fig_cel)
        
    with c_aud2:
        t_ton = np.linspace(0, 35, 500)
        local_pico = 35 * (1 - np.log10(freq_user/20) / np.log10(20000/20))
        envelope = (amp_user/120.0) * np.exp(-((t_ton - local_pico)**2)/10)
        
        fig_ton, ax_ton = plt.subplots(figsize=(10, 4))
        ax_ton.plot(t_ton, envelope, color='green', lw=2)
        ax_ton.fill_between(t_ton, envelope, color='green', alpha=0.3)
        ax_ton.set_title("Mecânica da Membrana Basilar (Onda Viajante)")
        ax_ton.set_xlabel("Distância do Estribo (mm) -> Direção ao Ápice")
        ax_ton.set_ylabel("Deslocamento")
        ax_ton.axvline(local_pico, color='black', linestyle='--', label=f'Pico em {local_pico:.1f} mm')
        ax_ton.legend()
        ax_ton.grid(alpha=0.3)
        st.pyplot(fig_ton)

    st.markdown("---")
    st.subheader("Microfisiologia: Transdução Mecanoelétrica (Célula Ciliada Interna)")
    st.markdown("""
    A variação mecânica da onda sonora gera movimentos oscilatórios contínuos nos estereocílios. 
    Quando defletidos mecanicamente para o lado do cinocílio, os canais iônicos se abrem, permitindo um massivo **influxo de Potássio (K+)** a partir da endolinfa, o que causa **Despolarização**. 
    O recuo da onda fecha os canais, interrompendo a corrente e gerando uma **Hiperpolarização** imediata. Note o comportamento AC (corrente alternada) do potencial de receptor.
    """)
    
    # Simulação do Potencial de Membrana oscilante da célula ciliada
    # Criamos tempo suficiente para visualizar 4 ciclos da frequência escolhida
    ciclos = 4
    duracao_ms = (ciclos / freq_user) * 1000 if freq_user > 0 else 10
    t_hc = np.linspace(0, duracao_ms, 500)
    
    # Onda sonora normalizada no tempo ajustado
    onda_estimulo = np.sin(2 * np.pi * (freq_user / 1000) * t_hc)
    
    # Vm responde de forma assimétrica (despolariza mais fortemente do que hiperpolariza)
    # O fator de amplitude é baseado no volume (dB) ajustado pelo usuário
    fator_amp = amp_user / 120.0
    vm_hc = -70.0 + (35.0 * onda_estimulo * (onda_estimulo > 0) * fator_amp) + (10.0 * onda_estimulo * (onda_estimulo <= 0) * fator_amp)
    
    fig_hc, ax_hc = plt.subplots(figsize=(12, 4))
    ax_hc.plot(t_hc, vm_hc, color='purple', lw=2.5)
    ax_hc.axhline(-70, color='black', linestyle='--', alpha=0.5, label='Repouso (-70 mV)')
    
    # Preenchimentos visuais para destacar a função do Potássio
    ax_hc.fill_between(t_hc, -70, vm_hc, where=(vm_hc > -70), color='red', alpha=0.3, label='Despolarização (Abertura: Influxo de K+)')
    ax_hc.fill_between(t_hc, -70, vm_hc, where=(vm_hc < -70), color='blue', alpha=0.3, label='Hiperpolarização (Fechamento: K+ bloqueado)')
    
    ax_hc.set_title(f"Potencial de Membrana do Receptor ({freq_user} Hz a {amp_user} dB)")
    ax_hc.set_xlabel("Tempo (ms)")
    ax_hc.set_ylabel("Potencial Intracelular Vm (mV)")
    ax_hc.grid(True, alpha=0.3)
    ax_hc.legend(loc='upper right')
    st.pyplot(fig_hc)

# ==========================================================
# ABA 7: EPILEPSIA (MICRO E MACRO)
# ==========================================================
with tabs[6]:
    st.header("🧠 Fisiopatologia: Epilepsia e Farmacologia")
    st.markdown("Estudo computacional dos padrões de disparo neuronal (Micro-sinal) e sua manifestação no Eletroencefalograma (Macro-sinal).")
    
    c_epi1, c_epi2 = st.columns(2)
    with c_epi1:
        tipo_crise = st.selectbox("Tipo de Atividade / Crise:", [
            "Cérebro Saudável (Controle)",
            "Epilepsia Crônica (Crise Focal)",
            "Status Epilepticus (Refratária)",
            "Crise de Ausência (Petit Mal)"
        ])
    with c_epi2:
        farmaco = st.selectbox("Intervenção Farmacológica:", [
            "Nenhum (Sem Medicação)",
            "Fenitoína (Bloqueador de Na+)",
            "Etossuximida (Bloqueador de Ca2+ Tipo-T)",
            "Diazepam (Agonista GABAérgico)"
        ])
        
    st.markdown("---")
    
    # Parâmetros base de Izhikevich
    a, b, c_param, d = 0.02, 0.2, -65, 8
    I_base = 10
    
    # Ajustes da Patologia (Dinâmica Matemática)
    if "Crônica" in tipo_crise:
        c_param, d = -50, 2  # Chattering / Bursting forte
        I_base = 15
    elif "Refratária" in tipo_crise:
        a, b, c_param, d = 0.1, 0.2, -65, 2 # Fast Spiking hiperativo
        I_base = 20
    elif "Ausência" in tipo_crise:
        a, b, c_param, d = 0.02, 0.25, -65, 0.05 # Resonador Talâmico
        I_base = 5 
        
    # Lógica de Intervenção Farmacológica (Erro Médico Simulado)
    alerta_erro = False
    if farmaco == "Fenitoína (Bloqueador de Na+)":
        if "Ausência" in tipo_crise:
            alerta_erro = True # Ausência não responde a Fenitoína!
        elif "Crônica" in tipo_crise or "Refratária" in tipo_crise:
            I_base = 2 # Diminui drasticamente a corrente, suprimindo o Na+
    elif farmaco == "Etossuximida (Bloqueador de Ca2+ Tipo-T)":
        if "Ausência" in tipo_crise:
            I_base = 2 # Sucesso! Bloqueia o circuito talamocortical
        elif "Crônica" in tipo_crise or "Refratária" in tipo_crise:
            alerta_erro = True # Fármaco ineficaz em crises não-ausência
    elif farmaco == "Diazepam (Agonista GABAérgico)":
        I_base -= 12 # Potencializa inibição globalmente
        if I_base < 0: I_base = 0
        
    if alerta_erro:
        st.error("⚠️ **ERRO CLÍNICO:** Este fármaco não é o indicado para esta crise! O mecanismo de ação não atinge o canal correto da fisiopatologia.")
    elif farmaco != "Nenhum (Sem Medicação)":
        st.success("✅ **FÁRMACO ATIVO:** Observe a supressão da crise no traçado eletrofisiológico e o restabelecimento do ritmo base.")
        
    # --- SIMULAÇÃO DE MICRO-ESCALA (Modelo de Izhikevich) ---
    T_sim = 1000 # 1 segundo (1000 ms)
    dt = 0.5
    t_steps = int(T_sim / dt)
    v = np.zeros(t_steps)
    u = np.zeros(t_steps)
    v[0] = -65
    u[0] = b * v[0]
    
    for i in range(1, t_steps):
        I_in = I_base
        # Se for ausência, o tálamo impulsiona uma onda senoidal lenta (~3 Hz) que ativa os canais T
        if "Ausência" in tipo_crise and I_base > 2:
            I_in = I_base + 8 * np.sin(2 * np.pi * 3 * (i * dt) / 1000)
            
        # Método de Euler para integração
        v_next = v[i-1] + dt * (0.04 * v[i-1]**2 + 5 * v[i-1] + 140 - u[i-1] + I_in)
        u_next = u[i-1] + dt * a * (b * v[i-1] - u[i-1])
        
        # Reset de disparo
        if v_next >= 30:
            v[i-1] = 30 # Força o pico visual
            v[i] = c_param
            u[i] = u_next + d
        else:
            v[i] = v_next
            u[i] = u_next

    # --- SIMULAÇÃO DE MACRO-ESCALA (Sintetizador EEG) ---
    t_eeg = np.linspace(0, 3, 3000) 
    eeg = np.zeros_like(t_eeg)
    
    if "Saudável" in tipo_crise:
        eeg = 2 * np.sin(2*np.pi*10*t_eeg) + np.random.normal(0, 0.5, len(t_eeg))
    elif "Crônica" in tipo_crise:
        if I_base > 5:
            # Mistura de ritmo basal com espículas focais intermitentes
            eeg = 15 * np.exp(-((t_eeg % 0.2 - 0.1)**2)/(2*0.01**2)) + 5 * np.sin(2*np.pi*5*t_eeg) + np.random.normal(0, 1, len(t_eeg))
        else: # Tratado
            eeg = 2 * np.sin(2*np.pi*10*t_eeg) + np.random.normal(0, 0.5, len(t_eeg))
    elif "Refratária" in tipo_crise:
        if I_base > 5:
            # Descargas elétricas de alta amplitude e frequência contínuas
            eeg = 20 * np.sin(2*np.pi*15*t_eeg) + 15 * np.sin(2*np.pi*25*t_eeg) + np.random.normal(0, 2, len(t_eeg))
        else: # Tratado (Diazepam reduz bastante a freq)
            eeg = 1 * np.sin(2*np.pi*4*t_eeg) + np.random.normal(0, 0.5, len(t_eeg)) 
    elif "Ausência" in tipo_crise:
        if I_base > 2:
            # Típico complexo Espícula-Onda (Spike and Wave) a 3 Hz
            onda = 15 * np.sin(2*np.pi*3*t_eeg)
            espícula = -25 * np.exp(-((t_eeg % (1/3) - 0.1)**2)/(2*0.005**2))
            eeg = onda + espícula + np.random.normal(0, 0.5, len(t_eeg))
        else: # Tratado (Etossuximida)
            eeg = 2 * np.sin(2*np.pi*10*t_eeg) + np.random.normal(0, 0.5, len(t_eeg))

    fig_epi, ax_epi = plt.subplots(2, 1, figsize=(12, 8))
    
    ax_epi[0].plot(np.arange(t_steps)*dt, v, color='purple', lw=1.5)
    ax_epi[0].set_title(f"Micro-sinal: Neurônio Isolado (Modelo de Izhikevich)")
    ax_epi[0].set_ylabel("Vm (mV)")
    ax_epi[0].set_xlim(0, 1000)
    ax_epi[0].grid(alpha=0.3)
    
    ax_epi[1].plot(t_eeg, eeg, color='black', lw=1.2)
    ax_epi[1].set_title(f"Macro-sinal: Registro de Campo Eletroencefalográfico (LFP)")
    ax_epi[1].set_xlabel("Tempo (s)")
    ax_epi[1].set_ylabel("Amplitude (µV)")
    ax_epi[1].set_xlim(0, 3)
    ax_epi[1].grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig_epi)
