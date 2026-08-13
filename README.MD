QuantGuard AI
Finanças quantitativas como camada de validação para análises produzidas por Inteligência Artificial.

Uma resposta financeira gerada por IA pode parecer convincente. Mas ela faz sentido quando colocada à prova por modelos quantitativos?**

O **QuantGuard AI** nasceu a partir dessa pergunta.

Modelos de linguagem conseguem produzir rapidamente análises sobre risco, alocação de portfólio, cenários de mercado e precificação de ativos. O problema é que uma resposta bem escrita não necessariamente é uma resposta financeiramente consistente.

Por isso, o objetivo deste projeto não é desenvolver mais um agente de IA para falar sobre mercado financeiro.

A proposta é fazer o caminho contrário:

**usar métodos quantitativos como benchmark para auditar aquilo que uma IA responde.**

I. A ideia

O projeto constrói uma carteira de ações brasileiras e acompanha um fluxo semelhante ao de uma análise financeira real:

**alocação → risco → stress → derivativos → validação da IA**

Para cada etapa, o QuantGuard utiliza modelos quantitativos formais para construir um resultado de referência.

Depois, esse resultado pode ser comparado ao output produzido por uma IA.

O sistema mede a diferença entre os dois e classifica o resultado em:

* **PASS** — resposta suficientemente próxima do benchmark;
* **REVIEW** — resultado plausível, mas que merece revisão;
* **FAIL** — divergência relevante e necessidade de revisão humana.

A ideia central é simples:

> IA pode auxiliar a análise financeira, mas resultados quantitativos importantes não deveriam ser aceitos apenas porque parecem plausíveis.


II. Portfolio Intelligence

A primeira etapa do projeto pergunta:

Como essa carteira deveria ser alocada considerando o trade-off entre risco e retorno?

São utilizados dados históricos de:

* PETR4
* VALE3
* ITUB4
* WEGE3

A partir deles, o projeto calcula retornos, volatilidades, covariâncias e simula **10.000 carteiras diferentes**.

São identificadas duas soluções importantes da Teoria Moderna de Portfólio:

Maximum Sharpe Portfolio: Carteira que busca o maior retorno ajustado ao risco.

Minimum Variance Portfolio: Carteira com a menor volatilidade possível dentro do universo analisado.

A Efficient Frontier permite visualizar como as diferentes combinações de ativos se distribuem entre risco e retorno.

Mas otimização histórica funciona no futuro? Esse é um ponto importante do projeto.

O QuantGuard não utiliza toda a base de dados para otimizar e depois avaliar a mesma carteira.

Os dados são separados em:

**Treino**

`2021 → 2024`

usado para encontrar os pesos ótimos.

**Teste**

`2025 → 2026`

usado para avaliar esses mesmos pesos em informações que o modelo não utilizou durante a otimização.

Isso permite uma **validação out-of-sample**.

As carteiras de Maximum Sharpe e Minimum Variance também são comparadas a um benchmark extremamente simples:

**Equal Weight — 25% em cada ativo.**

Essa comparação ajuda a mostrar um problema conhecido de modelos de otimização:

> uma carteira matematicamente ótima no passado não necessariamente permanecerá ótima no futuro.

---

# ⚠️ Risk Engine

Depois de escolher a carteira, a próxima pergunta é:

> **Quanto podemos perder?**

O QuantGuard calcula o risco da carteira por diferentes abordagens.

### Parametric VaR

Estima o Value at Risk assumindo uma distribuição paramétrica para os retornos.

### Historical VaR

Utiliza diretamente a distribuição observada dos retornos históricos.

### Expected Shortfall

O VaR informa um limite de perda associado a determinado nível de confiança.

O Expected Shortfall responde uma pergunta adicional:

> **quando ultrapassamos esse limite, qual é a perda média esperada?**

Isso permite observar melhor o comportamento da cauda da distribuição.

---

# 🧪 VaR Backtesting

Calcular VaR não é suficiente.

Também é necessário verificar se o modelo teria funcionado na prática.

Por isso, o QuantGuard utiliza o VaR estimado no período de treino e observa sua performance nos dados posteriores.

O sistema registra quantas vezes a perda real da carteira ultrapassou o VaR previsto.

Para um VaR de 95%, por exemplo, seria esperado que aproximadamente 5% das observações ultrapassassem o limite estimado.

Uma frequência de violações muito maior pode indicar:

**subestimação de risco pelo modelo.**

Essa etapa transforma o cálculo de risco em um problema de **validação de modelo**.

---

# 🌪️ Stress Testing

Mercados financeiros não se comportam sempre dentro das condições observadas em períodos normais.

Por isso, o projeto inclui cenários de stress.

## COVID-19 Crash

Os pesos da carteira atual são aplicados ao comportamento real dos mesmos ativos durante o crash de 2020.

A pergunta passa a ser:

> **O que teria acontecido com essa carteira se ela estivesse posicionada durante aquele período?**

O sistema calcula:

* impacto individual sobre os ativos;
* retorno da carteira;
* perda financeira estimada;
* valor final da carteira após o choque.

---

## Market Factor Stress

Também é calculada a sensibilidade histórica de cada ativo ao Ibovespa.

Para isso, é estimado o beta:

[
\beta_i = \frac{Cov(R_i,R_m)}{Var(R_m)}
]

A partir desses betas, o QuantGuard consegue simular um choque de mercado, como:

**Ibovespa -20%**

e estimar como cada ativo e a carteira completa tenderiam a reagir.

---

# 📈 Derivatives Engine

A etapa seguinte aborda precificação de derivativos.

O projeto calcula o preço de uma opção europeia utilizando dois métodos independentes:

### Black-Scholes

Solução analítica fechada para precificação da opção.

### Monte Carlo

Simulação do processo estocástico do preço do ativo e cálculo do payoff esperado descontado.

Os dois modelos são então comparados.

Na execução atual do projeto:

**Black-Scholes:** R$ 10,4506
**Monte Carlo:** R$ 10,4205
**Diferença:** aproximadamente **0,29%**

A convergência entre os dois métodos funciona como uma forma de **cross-model validation**.

---

# 🤖 AI Audit

Esta é a camada que conecta todo o projeto à Inteligência Artificial.

O QuantGuard pode receber um resultado produzido por uma IA e compará-lo ao benchmark quantitativo calculado pelo sistema.

Atualmente são avaliados três tipos de problema:

### Portfolio Allocation

A alocação sugerida pela IA é transformada em uma carteira e avaliada no período out-of-sample.

Em vez de simplesmente perguntar:

> “os pesos escolhidos pela IA são iguais aos do Markowitz?”

o sistema compara a consequência econômica das duas decisões por meio do **Sharpe Ratio**.

---

### Risk Estimation

A estimativa de VaR fornecida pela IA é comparada ao benchmark quantitativo.

O sistema calcula:

* erro percentual;
* score de precisão;
* classificação da resposta.

---

### Option Pricing

A IA recebe os mesmos parâmetros utilizados na precificação da opção.

Sua estimativa pode então ser comparada aos resultados de:

**Black-Scholes + Monte Carlo**

permitindo identificar divergências relevantes.

---

# 🛡️ Human Review

O objetivo do QuantGuard não é concluir que uma IA é simplesmente “boa” ou “ruim” em finanças.

A proposta é identificar **quando a resposta pode ser utilizada com confiança e quando deve existir revisão humana**.

O módulo de auditoria produz métricas como:

```text
Numerical Accuracy
Reliability Score
PASS / REVIEW / FAIL
Human Review Required
```

A versão atual utiliza a resposta da IA como uma entrada independente no sistema.

Essa separação é proposital: o modelo que está sendo auditado não recebe previamente o resultado quantitativo utilizado como benchmark.

---

# 🧠 Arquitetura

```text
Market Data
     ↓
Returns & Statistics
     ↓
┌─────────────────────────────┐
│ Portfolio Engine            │
│ Markowitz / Sharpe / MVP    │
└─────────────────────────────┘
     ↓
┌─────────────────────────────┐
│ Risk Engine                 │
│ VaR / ES / Backtesting      │
└─────────────────────────────┘
     ↓
┌─────────────────────────────┐
│ Stress Engine               │
│ Historical / Factor Stress  │
└─────────────────────────────┘
     ↓
┌─────────────────────────────┐
│ Derivatives Engine          │
│ Black-Scholes / Monte Carlo │
└─────────────────────────────┘
     ↓
┌─────────────────────────────┐
│ AI Audit                    │
│ Benchmark × AI Output       │
└─────────────────────────────┘
     ↓
       Dashboard
```

---

# 🖥️ Dashboard

O projeto possui uma interface desenvolvida em **Streamlit** dividida em cinco áreas:

### Portfolio

Efficient Frontier, composição das carteiras e validação out-of-sample.

### Risk

VaR, Expected Shortfall e backtesting.

### Stress Tests

Cenários históricos e choques de mercado.

### Derivatives

Comparação entre Black-Scholes e Monte Carlo.

### AI Audit

Avaliação dos outputs produzidos pela IA e Reliability Score.

> **Adicionar aqui uma imagem do dashboard**

---

# 🛠️ Tecnologias

**Python**

Bibliotecas principais:

* Pandas
* NumPy
* SciPy
* yfinance
* Plotly
* Streamlit

O projeto foi organizado em módulos independentes:

```text
src/
├── portfolio.py
├── risk.py
├── stress.py
├── derivatives.py
└── ai_audit.py
```

O `app.py` integra os diferentes motores quantitativos e apresenta os resultados no dashboard.

---

# ▶️ Executando o projeto

Clone o repositório:

```bash
git clone <URL-DO-REPOSITORIO>
```

Entre na pasta:

```bash
cd quantguard-ai
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute o dashboard:

```bash
streamlit run app.py
```

---

# 🔭 Próximos passos

O QuantGuard foi desenvolvido como um MVP e pode ser expandido em diferentes direções.

Entre elas:

* integração direta com APIs de modelos de linguagem;
* avaliação automática do raciocínio financeiro textual da IA;
* novos cenários macroeconômicos de stress;
* stress de juros com duration e convexidade;
* modelos adicionais de risco de cauda;
* Greeks e outras análises de derivativos;
* histórico de auditorias realizadas;
* comparação entre diferentes modelos de IA.

