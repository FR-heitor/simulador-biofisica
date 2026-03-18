# Desenvolvimento

Prof. Dr. Heitor Franco Santos - Departamento de Fisiologia - Universidade Federal de Sergipe

Esse aplicativo foi desenvolvido como parte de dúvidas, falta de ferramentas mais simples e do trabalho de monitoria exercido por ex-alunos do Centro de Ciências Biológicas e da Saúde da Universidade Federal de Sergipe. Ainda é um aplicativo em desenvolvimento então alguns equívocos ainda podem ser resolvidos. Importantes figuras históricas motivaram também o seu desenvolvimento, como os Professores Dr. Murilo Marchioro e Dr. Eduardo Antônio Conde Garcia que apresentavam brilhantemente esse conteúdo apenas com suas imponentes vozes e inteligência. Saudamos a essas brilhantes mentes nossa admiração.

De toda forma, a ferramente possui uma licença GNU, fique livre para adaptar e reportar qualquer melhoria e desenvolvimento aprimorado do sistema.

# Plataforma de simulação para Biofísica Celular

Bem-vindo(a) ao Laboratório Virtual de Bioeletrogênese da disciplina de **Biofísica**.

Esta plataforma foi desenvolvida para transformar fórmulas matemáticas em visualizações dinâmicas. Aqui, você poderá manipular o ambiente iônico de uma célula em tempo real e entender, na prática, como os tecidos biológicos geram e propagam eletricidade.

---

## A Plataforma

A interface está dividida em um painel lateral de **Controles** (onde você altera concentrações e permeabilidades) e uma tela principal com três abas de estudo:

### 1. 📊 Potenciais de Membrana e Equilíbrio (Nernst e GHK)
* **O que será apresentado:** Como o potencial de membrana em repouso ($V_m$) é um evento de ponderado, ou seja, possui dependências significativas ($E_{Na}$, $E_K$, $E_{Cl}$). 
* **Teste:** Aumente drasticamente a permeabilidade ao Sódio ($P_{Na}$) na barra lateral e veja o $V_m$ se aproximar do potencial de equilíbrio do Sódio, simulando o início de um potencial de ação.

### 2. 📈 Simulação Temporal e Correntes Iônicas
* **O que observar:** A evolução dinâmica da célula ao longo do tempo e o fluxo real de íons (corrente elétrica) atravessando a membrana.
* **Teste na prática:** Analise o gráfico de correntes. Note que a magnitude da corrente iônica ($I$) depende diretamente da Força Eletromotriz (a diferença entre o $V_m$ atual e o potencial de Nernst daquele íon).

### 3. ⚡ Dinâmica de Células Excitáveis (Potenciais de Ação)
Nesta aba, aplicamos estímulos elétricos virtuais para observar o comportamento de diferentes tecidos biológicos.
* **Neurônio:** Observe o clássico disparo rápido (~2 a 5 ms) com despolarização, repolarização e a fase de hiperpolarização pós-potencial.
* **Músculo Esquelético:** Note a diferença na velocidade. O potencial é ultrarrápido, permitindo contrações sequenciais rápidas sem um período refratário longo.
* **Músculo Cardíaco (Ventricular):** Observe a vital **Fase de Platô** (Fase 2). O gráfico se estende por quase 300 ms devido ao influxo prolongado de Cálcio ($Ca^{2+}$), o que previne a tetanização do coração e garante o bombeamento eficiente do sangue.

---

## 🛠️ Como Utilizar

1. O simulador roda diretamente no navegador, sem necessidade de instalação.
2. Utilize a barra lateral **🎛️ Manipulação de Variáveis** para ajustar:
   * **[Íon] Intracelular e Extracelular:** Concentrações em mM.
   * **Permeabilidade (P):** O quão abertos estão os canais para aquele íon.
3. Navegue pelas abas superiores para mudar o foco do seu estudo.
4. Na aba de **Células Excitáveis**, selecione o tecido desejado, ajuste a força do estímulo e clique em **Aplicar Estímulo** para gerar o gráfico. Se o estímulo for menor que o limiar, observe a resposta eletrotônica falha (tudo-ou-nada).

---

## 📚 Formulário de Referência

A matemática rodando nos bastidores da simulação:

* **Equação de Nernst (Equilíbrio de 1 íon):** $$E = \frac{RT}{zF} \ln\left(\frac{[C]_{out}}{[C]_{in}}\right)$$

* **Equação de Goldman-Hodgkin-Katz (GHK - Múltiplos íons):** $$V_m = \frac{RT}{F} \ln\left(\frac{P_K[K]_{out} + P_{Na}[Na]_{out} + P_{Cl}[Cl]_{in}}{P_K[K]_{in} + P_{Na}[Na]_{in} + P_{Cl}[Cl]_{out}}\right)$$
  *(Nota: O Cloreto possui valência negativa, invertendo sua posição na fração).*

* **Lei de Ohm para Membranas (Corrente Macroscópica):** $$I = g(V_m - E_{íon})$$

---
*Plataforma desenvolvida para o curso de Ciências Biológicas da Universidade Federal de Sergipe (CESAD/UFS). Coordenação: Prof. Dr. Heitor Franco Santos.*
