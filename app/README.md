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
Início
Simulador
Visão geral dos dados
Resultados dos modelos
Interpretação
Assistente LLM
Sobre e limitações
```

## Uso da Gemini API

O painel usa a Gemini API apenas como camada opcional de explicação textual.

A LLM não calcula probabilidades, não substitui o modelo supervisionado, não altera os resultados e não toma decisões sobre reclamações individuais.

A ordem de fallback configurada é:

```text
1. Gemini 3.5 Flash      -> gemini-3.5-flash
2. Gemini 3 Flash        -> gemini-3-flash
3. Gemini 3.1 Flash Lite -> gemini-3.1-flash-lite
```

## Configurando a chave da API

Defina a chave da Gemini API como variável de ambiente:

```bash
export GEMINI_API_KEY="SUA_CHAVE_AQUI"
```

Depois rode:

```bash
streamlit run app/streamlit_app.py
```

Também é possível usar o arquivo de segredos do Streamlit:

```bash
mkdir -p .streamlit
nano .streamlit/secrets.toml
```

Dentro do arquivo:

```toml
GEMINI_API_KEY = "SUA_CHAVE_AQUI"
```

## Funcionamento do fallback

Quando o usuário solicita uma explicação com LLM, o painel tenta chamar os modelos na seguinte ordem:

```text
gemini-3.5-flash
gemini-3-flash
gemini-3.1-flash-lite
```

Se o primeiro modelo falhar por indisponibilidade, limite ou erro de modelo, o painel tenta automaticamente o próximo. Se todos falharem, o painel mantém a explicação automática por template.

## Observação

O painel não refaz a limpeza dos dados e não treina os modelos a cada execução. Ele apenas carrega os artefatos já existentes no repositório.

As probabilidades estimadas não devem ser interpretadas como garantia de resolução para casos individuais.
