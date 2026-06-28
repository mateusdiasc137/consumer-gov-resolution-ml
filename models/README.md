# Modelos treinados

Esta pasta contém os modelos treinados no experimento de estimativa de resolução de reclamações do Consumidor.gov.br.

Os modelos foram salvos no formato `.joblib` usando pipelines do `scikit-learn`. Cada arquivo contém tanto o pré-processamento com `OneHotEncoder` quanto o classificador treinado.

## Arquivos

```text
models/
├── regressao_logistica.joblib
├── arvore_de_decisao.joblib
├── xgboost.joblib
├── svm_linear_calibrado.joblib
├── modelo_final.joblib
└── features.joblib
```

## Descrição dos modelos

| Arquivo | Descrição |
|---|---|
| `regressao_logistica.joblib` | Modelo de Regressão Logística treinado com `C=0.1`. Foi o melhor modelo em métricas probabilísticas. |
| `arvore_de_decisao.joblib` | Árvore de Decisão treinada com `max_depth=20` e `min_samples_leaf=500`. |
| `xgboost.joblib` | Modelo XGBoost treinado com a melhor configuração encontrada no tuning leve. Foi o melhor modelo classificatório. |
| `svm_linear_calibrado.joblib` | SVM Linear com calibração probabilística usando `CalibratedClassifierCV`. |
| `modelo_final.joblib` | Modelo escolhido para uso no painel probabilístico. Corresponde à Regressão Logística, pois apresentou menor Brier Score e menor Log Loss. |
| `features.joblib` | Lista de variáveis utilizadas na modelagem. |

## Modelo final

O arquivo principal para uso no painel é:

```text
modelo_final.joblib
```

Esse modelo foi escolhido com base na qualidade das probabilidades estimadas, e não apenas no desempenho classificatório. Embora o XGBoost tenha apresentado maior F1 macro e maior ROC-AUC, a Regressão Logística obteve menor Brier Score e menor Log Loss, sendo mais adequada para o simulador de probabilidade.

## Como carregar o modelo

Exemplo de uso em Python:

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

## Observação

As probabilidades estimadas não devem ser interpretadas como garantia de resolução para reclamações individuais. O modelo aprende padrões estatísticos a partir de dados históricos e deve ser usado apenas para análise exploratória e comunicação de dados.
