# Desenvolvimento

Prof. Dr. Heitor Franco Santos - Departamento de Fisiologia - Universidade Federal de Sergipe

O Simulador de fenômenos biofísicos celulares é um aplicativo desenvolvido pelo Professor Heitor Franco do Departamento de Fisiologia da Universidade Federal de Sergipe, como parte de dúvidas e falta de ferramentas mais simples que permitam a visualização de fenômenos da eletrofisiologia celular. Esse aplicativo vem da iniciativa do Professor Heitor com apoio dos trabalho de monitoria desenvolvido por ex-alunos do Centro de Ciências Biológicas e da Saúde da Universidade Federal de Sergipe. Ainda é um aplicativo em desenvolvimento então alguns equívocos ainda podem ser resolvidos. Importantes figuras históricas motivaram também o seu desenvolvimento, como os Professores Dr. Murilo Marchioro e Dr. Eduardo Antônio Conde Garcia que apresentavam brilhantemente esse conteúdo apenas com suas imponentes vozes e inteligência. Saudamos a essas brilhantes mentes nossa admiração.

De toda forma, a ferramenta é para uso e você pode reportar qualquer melhoria e desenvolvimento aprimorado do sistema.

# Plataforma de simulação

## 📖 Sobre o Projeto
O **Simulador de Biofísica UFS** é uma plataforma computacional interativa desenvolvida para o ensino e pesquisa em Fisiologia e Biofísica. Utilizando equações diferenciais e modelos matemáticos validados na literatura científica mundial, a aplicação traduz fenômenos biológicos complexos em gráficos dinâmicos e interativos, executados em tempo real pelo navegador.

A plataforma cobre desde a microfisiologia celular (canais iônicos, repouso, propriedades passivas) até a macrofisiologia sistêmica (hemodinâmica, biomecânica muscular e registros eletrofisiológicos clínicos).

---

⚙️ Módulos, Modelos Matemáticos e Referências

1. 📊 Potencial de Repouso (Estado Eletroquímico)
*   **Descrição:** Calcula o potencial de equilíbrio de íons individuais e o potencial de membrana global em repouso com base nas concentrações intra e extracelulares.
*   **Modelos Computacionais:** Equação de Nernst e Equação de Goldman-Hodgkin-Katz (GHK).
*   **Referências:** 
    *   Curi, R., & Procopio, J. (2017). *Fisiologia Básica* (2ª Ed.). Guanabara Koogan. (Cap. 1: Biofísica das membranas excitáveis).
    *   Hodgkin, A. L., & Horowicz, P. (1959). *The influence of potassium and chloride ions on the membrane potential of single muscle fibres*.

2. 🔌 Comportamento Elétrico Passivo (Circuito RC)
*   **Descrição:** Modela a membrana celular como um circuito elétrico (Resistor-Capacitor em paralelo) submetido a pulsos retangulares de corrente.
*   **Modelos Computacionais:** Dinâmica de Carga e Descarga Exponencial ($V(t) = V_{max}(1 - e^{-t/\tau})$). A membrana atua como dielétrico e as proteínas como resistores.
*   **Referências:**
    *   Okuno, E., Caldas, I. L., & Chow, C. (1986). *Física para Ciências Biológicas e Biomédicas*. Harbra.
    *   Singer, S. J., & Nicolson, G. L. (1972). *The fluid mosaic model of the structure of cell membranes*. Science.

3. ⚡ Células Excitáveis e Modulação Autonômica
*   **Descrição:** Simulação do Potencial de Ação em neurônios, miócitos esqueléticos e cardíacos, com adição da resposta farmacológica (TTX, TEA, Verapamil) e modulação autonômica.
*   **Modelos Computacionais:** Equações de Hodgkin-Huxley (resolvidas via `scipy.integrate.solve_ivp`); Integração para simulação do Platô Cardíaco e Acoplamento Excitação-Contração.
*   **Referências:**
    *   Hodgkin, A. L., & Huxley, A. F. (1952). *A quantitative description of membrane current...* J Physiol.
    *   Guyton, A. C., & Hall, J. E. (2016). *Tratado de Fisiologia Médica*. Elsevier.

4. 🫀 Sinais Macroscópicos (ECG, EEG, EMG)
*   **Descrição:** Síntese de biossinais clínicos a partir de princípios bioelétricos de campo de superfície. 
*   **Modelos Computacionais:** 
    *   **ECG:** Superposição de ondas Gaussianas com variação dromotrópica temporal.
    *   **EEG:** Rede Neural simulando a Codificação Preditiva e Habituação do ERP.
    *   **EMG:** Modelo de Potencial de Ação Muscular Composto (CMAP).
*   **Referências:**
    *   Friston, K. (2005). *A theory of cortical responses*.
    *   Okuno, E., et al. (1986). *Física para Ciências Biológicas e Biomédicas*.

5. 👁️ Óptica Fisiológica e Fototransdução
*   **Descrição:** Comportamento quântico dos fotorreceptores e a resposta elétrica hiperpolarizante à luz.
*   **Modelos Computacionais:** Espectro de absorção Gaussiano de Opsinas e cinética de decaimento do Vm por interrupção da Corrente de Escuro.
*   **Referências:**
    *   Curi, R., & Procopio, J. (2017). *Fisiologia Básica* (Cap. 11: Bases da Fisiologia Sensorial - Silveira, L. C. L.).

6. 👂 Acústica Fisiológica e Mecanotransdução
*   **Descrição:** Tonotopia coclear, potenciais microfônicos e a codificação não-linear de intensidade sonora.
*   **Modelos Computacionais:** Envelope vetorial da onda viajante; Função sigmoide de saturação de disparo da fibra nervosa (~300 impulsos/s); Soma de potenciais microfônicos.
*   **Referências:**
    *   Tasaki, I. et al. (1952). *The space-time pattern of the cochlear microphonics*.
    *   Curi, R., & Procopio, J. (2017). *Fisiologia Básica* (Cap. 11: Audição).

7. 🧠 Epilepsia Computacional e Farmacologia
*   **Descrição:** Simulação da fisiopatologia das crises epilépticas (Focal, Refratária, Ausência) e testagem de farmacodinâmica direcional.
*   **Modelos Computacionais:** Modelo Neuronal de Izhikevich (Resonadores e Integradores) para *bursting* e *fast-spiking*; Sintetizador do complexo Espícula-Onda (3 Hz).
*   **Referências:**
    *   Izhikevich, E. M. (2003). *Simple model of spiking neurons*. IEEE.

8. 💪 Biomecânica Muscular
*   **Descrição:** Integração da microanatomia com a cinemática do movimento do músculo esquelético.
*   **Modelos Computacionais:** Relação Comprimento-Tensão (actina-miosina) e Equação de Força-Velocidade de Hill (Isotônica vs Isométrica).
*   **Referências:**
    *   Gordon, A. M., et al. (1966). *The variation in isometric tension with sarcomere length*.
    *   Curi, R., & Procopio, J. (2017). *Fisiologia Básica* (Cap. 9: Fisiologia Muscular).

9. 🩸 Hemodinâmica, Mecânica Cardíaca e Covariância Sistêmica
*   **Descrição:** Estudo aprofundado do fluxo sanguíneo, dinâmica do coração como bomba, resistência vascular periférica e a entrega tecidual de oxigênio em condições basais e patológicas (como a insuficiência cardíaca por hipertrofia).
*   **Modelos Computacionais:** 
    *   **Mecânica Cardíaca (A Bomba):** Cálculo do Débito Cardíaco ($DC = FC \times VS$) e determinação da Fração de Ejeção ($FE = VS/VDF$) baseada no Mecanismo de Frank-Starling e no inotropismo ventricular.
    *   **Resistência e Pressão (Os Vasos):** Aplicação direta da Lei de Poiseuille ($R \propto 1/r^4$) e cálculo da Pressão Arterial Média analógica à Lei de Ohm ($PAM = DC \times RPT$).
    *   **Oxigenação Tecidual:** Equação de transporte de $O_2$ acoplada ao volume minuto cardíaco (baseado no Princípio de Fick) e visualização vetorial da Cascata Pressórica ao longo do leito vascular (Aorta até Veias Cavas).
**Referências:**
    *   Salgado, H. C. (2017). *Hemodinâmica*. Em Curi, R., & Procopio, J. *Fisiologia Básica*.
    *   Lopes, O. U. (2017). *O Coração como Bomba | O Ciclo Cardíaco*. Em Curi, R., & Procopio, J. *Fisiologia Básica*.
    *   Rossoni, L. V. (2017). *Microcirculação e Mecanismos de Controle do Fluxo Sanguíneo*. Em Curi, R., & Procopio, J. *Fisiologia Básica*.
    *   Okuno, E., et al. (1986). *Física para Ciências Biológicas e Biomédicas* (Unidade IV - Fluidos em Sistemas Biológicos).



🚀 Como Executar o Projeto Localmente

1. **Pré-requisitos:** Certifique-se de ter o Python 3.9 (ou superior) instalado.
2. **Instalação das Dependências:**
   Abra o terminal e instale as bibliotecas necessárias utilizando o `pip`:
   

   pip install streamlit numpy scipy matplotlib

---
*Plataforma desenvolvida para uso em disciplinas de Biofísica e Fisiologia - UFS. Coordenação: Prof. Dr. Heitor Franco Santos.*
