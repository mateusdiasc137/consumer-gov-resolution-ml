# Painel Inteligente para Análise e Estimativa de Resolução de Reclamações no Consumidor.gov.br

Este repositório contém o código, os dados processados, os modelos treinados, os resultados e as instruções de reprodução do projeto final da disciplina **INF 420 - Inteligência Artificial I**.

O objetivo do projeto é avaliar modelos supervisionados de aprendizado de máquina para estimar a probabilidade de resolução de reclamações registradas no **Consumidor.gov.br**.

## Autores

- Mateus José Dias
- Lucas Carvalho de Góes

## Visão geral do projeto

O trabalho utiliza dados públicos do Consumidor.gov.br para formular uma tarefa de classificação binária:

- `Resolvida` → classe positiva (`1`)
- `Não Resolvida` → classe negativa (`0`)

Foram consideradas apenas reclamações finalizadas e avaliadas explicitamente pelo consumidor. Reclamações não avaliadas, canceladas ou encerradas sem avaliação explícita foram removidas da tarefa principal.

Foram comparadas quatro famílias de modelos:

- Regressão Logística
- Árvore de Decisão
- XGBoost
- SVM Linear calibrado

Também foram avaliados dois baselines estatísticos:

- média geral de resolução;
- média histórica por grupo de problema.

## Estrutura do repositório

```text
consumer-gov-resolution-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── 01_limpar_dados.py
│   ├── 01_conferir_parquets.py
│   ├── 02_criar_conjuntos_modelagem.py
│   ├── 02b_criar_train_full.py
│   ├── 03_tuning_leve.py
│   ├── 03_treinar_modelos.py
│   ├── 04_ver_resultados_completos.py
│   ├── 05_gerar_graficos_resultados.py
│   ├── 06_curva_calibracao.py
│   └── 07_interpretar_variaveis.py
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   ├── catalogo_processamento.csv
│   │   └── parts/
│   │       └── *.parquet
│   └── modelagem/
│       ├── train.parquet
│       ├── valid.parquet
│       └── test.parquet
├── results/
│   ├── resultados_validacao.csv
│   ├── resultados_teste.csv
│   ├── tuning_validacao.csv
│   ├── melhores_configs_classificacao.csv
│   ├── melhores_configs_probabilidade.csv
│   ├── graficos/
│   └── interpretacao/
├── models/
│   ├── README.md
│   ├── regressao_logistica.joblib
│   ├── arvore_de_decisao.joblib
│   ├── xgboost.joblib
│   ├── svm_linear_calibrado.joblib
│   ├── modelo_final.joblib
│   └── features.joblib
└── app/
    ├── README.md
    └── streamlit_app.py
```

## Dados

Os dados brutos utilizados são públicos e foram obtidos no portal de dados abertos do Consumidor.gov.br.

Os arquivos CSV brutos originais **não estão incluídos** neste repositório. Para reproduzir o processamento desde o início, baixe os arquivos da base completa do Consumidor.gov.br e coloque-os em:

```text
data/raw/
```

A estrutura esperada é semelhante a:

```text
data/raw/basecompleta2014.csv
data/raw/basecompleta2015.csv
data/raw/basecompleta2016.csv
...
data/raw/basecompleta2026-05.csv
```

## Dados incluídos no pacote

Este repositório inclui os dados processados necessários para reproduzir a modelagem sem baixar novamente os CSVs brutos:

```text
data/processed/
data/modelagem/
```

A pasta `data/processed/` contém os arquivos Parquet gerados após a limpeza dos dados, além do arquivo:

```text
data/processed/catalogo_processamento.csv
```

Esse catálogo registra informações sobre os arquivos processados, como arquivos de origem, quantidade de linhas e etapas de processamento.

A pasta `data/modelagem/` contém os conjuntos finais usados nos experimentos:

```text
data/modelagem/train.parquet
data/modelagem/valid.parquet
data/modelagem/test.parquet
```

A separação dos dados foi temporal:

- treino: 2014 a 2024;
- validação: 2025;
- teste: 2026.

## Ambiente de execução

Recomenda-se utilizar Python 3.10 ou superior.

Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Um `requirements.txt` mínimo para o projeto é:

```text
pandas
numpy
duckdb
pyarrow
scikit-learn
xgboost
matplotlib
joblib
streamlit
plotly
requests
```

## Reprodução rápida

Como os conjuntos de modelagem e os modelos treinados já estão incluídos no repositório, é possível reproduzir os principais resultados sem executar novamente toda a limpeza dos dados brutos.

Para reproduzir os resultados finais a partir dos arquivos em `data/modelagem/`, execute:

```bash
python src/03_treinar_modelos.py
python src/04_ver_resultados_completos.py
python src/05_gerar_graficos_resultados.py
python src/06_curva_calibracao.py
python src/07_interpretar_variaveis.py
```

## Reprodução completa

Para reproduzir o experimento desde os CSVs brutos, baixe os dados originais do Consumidor.gov.br, coloque os arquivos em `data/raw/` e execute:

```bash
python src/01_limpar_dados.py
python src/01_conferir_parquets.py
python src/02_criar_conjuntos_modelagem.py
python src/03_tuning_leve.py
python src/03_treinar_modelos.py
python src/04_ver_resultados_completos.py
python src/05_gerar_graficos_resultados.py
python src/06_curva_calibracao.py
python src/07_interpretar_variaveis.py
```

## Descrição dos scripts

### `01_limpar_dados.py`

Lê os arquivos CSV brutos do Consumidor.gov.br, trata diferenças de codificação, separadores, nomes de colunas e formatos de data. Também filtra reclamações finalizadas e avaliadas e cria a variável alvo `target_resolvida`.

### `01_conferir_parquets.py`

Confere os arquivos Parquet gerados após a limpeza. Produz contagens gerais, distribuição da classe alvo e estatísticas exploratórias básicas.

### `02_criar_conjuntos_modelagem.py`

Cria os conjuntos de treino, validação e teste usados na modelagem, usando separação temporal.

### `02b_criar_train_full.py`

Gera um conjunto de treino completo com todos os registros disponíveis entre 2014 e 2024. Esse script é opcional e pode ser usado para treinar uma versão final do modelo probabilístico com toda a base histórica.

### `03_tuning_leve.py`

Executa uma busca leve de hiperparâmetros usando o conjunto de validação. São testadas configurações diferentes para cada família de modelo.

### `03_treinar_modelos.py`

Treina os modelos finais, avalia os modelos nos conjuntos de validação e teste, calcula as métricas e salva os resultados.

### `04_ver_resultados_completos.py`

Imprime as tabelas completas de resultados de validação e teste.

### `05_gerar_graficos_resultados.py`

Gera gráficos comparativos de desempenho, incluindo:

- F1 macro;
- ROC-AUC;
- Brier Score;
- Log Loss;
- matriz de confusão do XGBoost;
- matriz de confusão da Regressão Logística.

### `06_curva_calibracao.py`

Gera a curva de calibração do modelo final usado no painel probabilístico.

### `07_interpretar_variaveis.py`

Gera análises de importância das variáveis para a Regressão Logística e para o XGBoost.

## Modelos avaliados

Foram comparadas quatro famílias de modelos:

| Modelo | Família |
|---|---|
| Regressão Logística | Linear / probabilística |
| Árvore de Decisão | Árvores |
| XGBoost | Ensemble / Boosting |
| SVM Linear calibrado | SVM |

A Regressão Logística foi escolhida como modelo principal do painel probabilístico por apresentar melhores métricas probabilísticas. O XGBoost foi o melhor modelo em termos de classificação.

## Baselines

Foram utilizados dois baselines:

| Baseline | Descrição |
|---|---|
| Média geral | Usa a taxa média de resolução observada no treino |
| Média por Grupo Problema | Usa a taxa histórica de resolução por grupo de problema |

Esses baselines foram usados para verificar se os modelos de aprendizado de máquina superam regras estatísticas simples.

## Principais resultados

No conjunto de teste de 2026, os principais resultados foram:

| Modelo | F1 macro | ROC-AUC | Brier Score | Log Loss |
|---|---:|---:|---:|---:|
| XGBoost | 0.640 | 0.690 | 0.229 | 0.651 |
| Regressão Logística | 0.634 | 0.679 | 0.214 | 0.619 |
| SVM Linear calibrado | 0.630 | 0.675 | 0.231 | 0.655 |
| Árvore de Decisão | 0.626 | 0.648 | 0.220 | 0.632 |
| Baseline por Grupo Problema | 0.458 | 0.488 | 0.281 | 0.756 |
| Baseline média geral | 0.386 | 0.500 | 0.280 | 0.755 |

O XGBoost apresentou o melhor desempenho classificatório, com maior F1 macro e maior ROC-AUC.

A Regressão Logística apresentou as melhores métricas probabilísticas, com menor Brier Score e menor Log Loss, sendo escolhida como modelo principal para o simulador de probabilidade.

## Resultados gerados

Os principais arquivos de saída ficam em:

```text
results/
```

Arquivos esperados:

```text
results/resultados_validacao.csv
results/resultados_teste.csv
results/tuning_validacao.csv
results/melhores_configs_classificacao.csv
results/melhores_configs_probabilidade.csv
results/graficos/
results/interpretacao/
```

## Modelos treinados

Os modelos treinados estão incluídos em:

```text
models/
```

Arquivos incluídos:

```text
models/regressao_logistica.joblib
models/arvore_de_decisao.joblib
models/xgboost.joblib
models/svm_linear_calibrado.joblib
models/modelo_final.joblib
models/features.joblib
```

O arquivo principal para uso no painel probabilístico é:

```text
models/modelo_final.joblib
```

Esse modelo corresponde à Regressão Logística, escolhida por apresentar melhor desempenho probabilístico no conjunto de teste.

## Como carregar o modelo final

Exemplo simples de uso em Python:

```python
import joblib
import pandas as pd

modelo = joblib.load("models/modelo_final.joblib")

exemplo = pd.DataFrame([{
    "Canal de Origem": "Internet",
    "Região": "Sudeste",
    "UF": "MG",
    "Sexo": "M",
    "Faixa Etária": "entre 31 a 40 anos",
    "Ano Abertura": "2026",
    "Mês Abertura": "5",
    "Segmento de Mercado": "Energia Elétrica",
    "Área": "Telecomunicações",
    "Assunto": "Serviços de telefonia",
    "Grupo Problema": "Cobrança / Contestação",
    "Problema": "Cobrança indevida / abusiva",
    "Como Comprou Contratou": "Internet",
    "Procurou Empresa": "S"
}])

probabilidade = modelo.predict_proba(exemplo)[:, 1][0]

print(f"Probabilidade estimada de resolução: {probabilidade:.2%}")
```

## Painel interativo

O repositório também inclui um painel interativo desenvolvido em **Streamlit**, localizado na pasta:

```text
app/
```

A estrutura da pasta é:

```text
app/
├── README.md
└── streamlit_app.py
```

O painel utiliza os artefatos gerados neste projeto para permitir a exploração dos dados, visualização dos resultados e simulação da probabilidade estimada de resolução de uma reclamação.

Os principais arquivos utilizados pelo painel são:

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

Para executar o painel, utilize:

```bash
streamlit run app/streamlit_app.py
```

Instruções específicas sobre a estrutura interna do painel, uso da LLM opcional e detalhes de execução estão disponíveis no arquivo:

```text
app/README.md
```

## Observações éticas

As probabilidades estimadas pelo modelo não devem ser interpretadas como garantia para casos individuais. O modelo aprende associações estatísticas a partir de dados históricos e deve ser usado apenas para análise exploratória e comunicação de dados.

O projeto não substitui órgãos de defesa do consumidor, aconselhamento jurídico ou indicadores oficiais do Consumidor.gov.br.

## Como citar a fonte dos dados

Os dados utilizados são provenientes do portal de dados abertos do Consumidor.gov.br. Recomenda-se citar também o dicionário de dados oficial da plataforma no relatório e em trabalhos derivados.

## Licença

Este repositório é destinado a fins acadêmicos e educacionais. Os dados brutos pertencem à fonte pública original e devem ser utilizados conforme os termos e políticas do Consumidor.gov.br.
