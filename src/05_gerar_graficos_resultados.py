from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


PASTA_RESULTADOS = Path("data/resultados")
PASTA_GRAFICOS = PASTA_RESULTADOS / "graficos"
PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

ARQ_TESTE = PASTA_RESULTADOS / "resultados_teste.csv"

df_teste = pd.read_csv(ARQ_TESTE)


def grafico_barras(df, metrica, titulo, arquivo, menor_melhor=False):
    """
    Gera gráfico de barras horizontais para uma métrica.
    menor_melhor=True para métricas como Brier Score e Log Loss.
    """

    dados = df.copy()

    if menor_melhor:
        dados = dados.sort_values(metrica, ascending=False)
    else:
        dados = dados.sort_values(metrica, ascending=True)

    plt.figure(figsize=(10, 5))
    plt.barh(dados["modelo"], dados[metrica])
    plt.xlabel(metrica)
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / arquivo, dpi=200)
    plt.close()


def matriz_confusao_modelo(df, nome_modelo, arquivo_saida):
    """
    Gera matriz de confusão para um modelo específico.
    Usa as colunas já salvas em resultados_teste.csv.
    """

    linha = df[df["modelo"] == nome_modelo]

    if linha.empty:
        raise ValueError(f"Modelo não encontrado no CSV: {nome_modelo}")

    linha = linha.iloc[0]

    tn = int(linha["tn_nao_resolvida"])
    fp = int(linha["fp_nao_resolvida_classificada_como_resolvida"])
    fn = int(linha["fn_resolvida_classificada_como_nao_resolvida"])
    tp = int(linha["tp_resolvida"])

    matriz = np.array([
        [tn, fp],
        [fn, tp]
    ])

    total = matriz.sum()

    plt.figure(figsize=(7, 6))
    plt.imshow(matriz)

    plt.title(f"Matriz de confusão - {nome_modelo}")
    plt.xticks(
        [0, 1],
        ["Prev. Não Resolvida", "Prev. Resolvida"],
        rotation=20
    )
    plt.yticks(
        [0, 1],
        ["Real Não Resolvida", "Real Resolvida"]
    )

    for i in range(2):
        for j in range(2):
            valor = matriz[i, j]
            percentual = valor / total * 100
            texto = f"{valor:,}\n({percentual:.1f}%)".replace(",", ".")
            plt.text(
                j,
                i,
                texto,
                ha="center",
                va="center"
            )

    plt.xlabel("Classe prevista")
    plt.ylabel("Classe real")
    plt.tight_layout()
    plt.savefig(PASTA_GRAFICOS / arquivo_saida, dpi=200)
    plt.close()


# ============================
# Gráficos de comparação geral
# ============================

grafico_barras(
    df_teste,
    "f1_macro",
    "Comparação de F1 macro no teste",
    "f1_macro_teste.png",
    menor_melhor=False
)

grafico_barras(
    df_teste,
    "roc_auc",
    "Comparação de ROC-AUC no teste",
    "roc_auc_teste.png",
    menor_melhor=False
)

grafico_barras(
    df_teste,
    "brier_score",
    "Comparação de Brier Score no teste",
    "brier_score_teste.png",
    menor_melhor=True
)

grafico_barras(
    df_teste,
    "log_loss",
    "Comparação de Log Loss no teste",
    "log_loss_teste.png",
    menor_melhor=True
)


# ============================
# Matrizes de confusão
# ============================

matriz_confusao_modelo(
    df_teste,
    "XGBoost",
    "matriz_confusao_xgboost.png"
)

matriz_confusao_modelo(
    df_teste,
    "Regressão Logística",
    "matriz_confusao_regressao_logistica.png"
)


print("Gráficos salvos em:")
print(PASTA_GRAFICOS)

print("\nArquivos gerados:")
for arquivo in sorted(PASTA_GRAFICOS.glob("*.png")):
    print("-", arquivo)
