# Painel interativo

Esta pasta contém o painel interativo do projeto de estimativa de resolução de reclamações no Consumidor.gov.br.

O painel foi desenvolvido com **Streamlit** e utiliza os artefatos já gerados no projeto: modelos treinados, dados de modelagem, resultados, gráficos e arquivos de interpretação.

## Estrutura

```text
app/
├── README.md
└── streamlit_app.py
```

## Como executar

A partir da raiz do repositório, instale as dependências:

```bash
pip install -r requirements.txt
```

Depois execute:

```bash
streamlit run app/streamlit_app.py
```

## Arquivos usados pelo painel

O painel espera encontrar os seguintes arquivos na raiz do repositório:

```text
models/modelo_final.joblib
models/features.joblib
data/modelagem/train.parquet
data/modelagem/valid.parquet
data/modelagem/test.parquet
results/resultados_validacao.csv
results/resultados_teste.csv
results/graficos/
results/interpretacao/
```

O modelo usado no simulador probabilístico é:

```text
models/modelo_final.joblib
```

Esse modelo corresponde à Regressão Logística, escolhida por apresentar melhor desempenho probabilístico no conjunto de teste.

A classe positiva do modelo é:

```text
target_resolvida = 1
```

ou seja, reclamação avaliada como resolvida.

A probabilidade exibida pelo painel é calculada com:

```python
modelo.predict_proba(entrada)[:, 1][0]
```

## Páginas do painel

O painel possui as seguintes páginas:

```text
Início              → Visão geral do projeto, KPIs e navegação rápida
Simulador           → Preenchimento manual ou por IA + cálculo de probabilidade
Visão geral         → Distribuições dos dados por segmento, grupo, UF e ano
Resultados          → Comparação entre modelos (F1, ROC-AUC, Brier, Log Loss)
Interpretação       → Importância das variáveis e categorias por modelo
Sobre e limitações  → Escopo, cuidados éticos e status dos arquivos
```

## Simulador: preenchimento automático com IA

O simulador possui um campo de texto onde o usuário pode descrever sua reclamação em linguagem natural. Ao clicar em **"🤖 Preencher com IA"**, a descrição é enviada à Gemini API, que extrai as informações e seleciona automaticamente os valores mais adequados para cada campo do formulário.

### Exemplo de uso

O usuário digita:

> Em março de 2026, comprei um celular pela internet em uma loja de eletrônicos. O produto chegou com defeito e, mesmo após procurar a empresa, não consegui trocar. Moro em São Paulo.

A IA preenche automaticamente campos como:

- **Mês Abertura** → `3`
- **Ano Abertura** → `2026`
- **Canal de Origem** → `Internet`
- **Segmento de Mercado** → segmento mais próximo de eletrônicos
- **Problema** → problema mais próximo de troca/defeito
- **Procurou Empresa** → `S`
- **UF** → `SP`
- **Região** → `Sudeste`

### Como funciona internamente

1. A função `extrair_campos_com_gemini()` monta um prompt contendo a descrição do usuário e todas as opções válidas de cada campo (extraídas dos dados de treino).
2. A Gemini API retorna um JSON estruturado com os valores selecionados.
3. Os valores são validados contra as opções reais do dataset (com fallback case-insensitive). Valores inválidos são substituídos por "Não informado".
4. Os selectboxes do formulário são pré-preenchidos via `st.session_state`.
5. O usuário pode **modificar qualquer campo manualmente** antes de clicar "Estimar probabilidade".

## Uso da Gemini API

O painel usa a Gemini API em dois contextos distintos. Em nenhum deles a LLM calcula probabilidades, altera resultados ou substitui o modelo supervisionado.

### 1. Preenchimento automático de campos (Simulador)

Quando o usuário descreve sua reclamação em texto livre, a IA extrai informações estruturadas e preenche o formulário. A resposta é um JSON com os valores de cada campo, validados contra as opções reais do dataset.

### 2. Explicação personalizada do resultado (Simulador)

Após o cálculo da probabilidade, o usuário pode opcionalmente solicitar uma explicação em linguagem simples gerada pela IA. O prompt inclui:

- Todas as 14 características da reclamação simulada
- Probabilidade estimada e comparação com médias (geral e do grupo)
- Dados de interpretação do modelo (quais categorias da reclamação aumentam ou reduzem a probabilidade segundo a Regressão Logística)

Isso gera uma explicação específica para cada caso, e não uma resposta genérica.

## Configurando a chave da API

A chave da Gemini API é necessária apenas para as funcionalidades de IA. O painel funciona normalmente sem ela — apenas o preenchimento automático e a explicação por IA ficam indisponíveis.

Defina a chave como variável de ambiente:

```bash
export GEMINI_API_KEY="SUA_CHAVE_AQUI"
streamlit run app/streamlit_app.py
```

Ou use o arquivo de segredos do Streamlit:

```bash
mkdir -p .streamlit
```

Dentro de `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "SUA_CHAVE_AQUI"
```

## Fallback entre modelos Gemini

Quando qualquer funcionalidade de IA é acionada, o painel tenta os modelos na seguinte ordem:

```text
1. gemini-3.5-flash
2. gemini-3.1-flash-lite
3. gemini-2.5-flash
```

Se o primeiro modelo falhar por indisponibilidade, limite de requisições ou erro, o painel tenta automaticamente o próximo. Se todos falharem:

- No preenchimento automático: uma mensagem de erro é exibida e o formulário permanece sem alterações.
- Na explicação do resultado: a explicação automática por template (sem IA) é exibida no lugar.

## Observação

O painel não refaz a limpeza dos dados e não treina os modelos a cada execução. Ele apenas carrega os artefatos já existentes no repositório.

As probabilidades estimadas não devem ser interpretadas como garantia de resolução para casos individuais.
