from pathlib import Path
from textwrap import shorten
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt


PASTA_MODELOS = Path("models")
PASTA_RESULTADOS = Path("data/resultados")
PASTA_GRAFICOS = PASTA_RESULTADOS / "graficos"
PASTA_INTERPRETACAO = PASTA_RESULTADOS / "interpretacao"

PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)
PASTA_INTERPRETACAO.mkdir(parents=True, exist_ok=True)

ARQ_REGRESSAO = PASTA_MODELOS / "regressao_logistica.joblib"
ARQ_XGBOOST = PASTA_MODELOS / "xgboost.joblib"


VARIAVEIS_ORIGINAIS = [
    "Canal de Origem",
    "Região",
    "UF",
    "Sexo",
    "Faixa Etária",
    "Ano Abertura",
    "Mês Abertura",
    "Segmento de Mercado",
    "Área",
    "Assunto",
    "Grupo Problema",
    "Problema",
    "Como Comprou Contratou",
    "Procurou Empresa",
]


VARIAVEIS_ABREV = {
    "Canal de Origem": "Canal",
    "Região": "Região",
    "UF": "UF",
    "Sexo": "Sexo",
    "Faixa Etária": "Faixa etária",
    "Ano Abertura": "Ano",
    "Mês Abertura": "Mês",
    "Segmento de Mercado": "Segmento",
    "Área": "Área",
    "Assunto": "Assunto",
    "Grupo Problema": "Grupo",
    "Problema": "Problema",
    "Como Comprou Contratou": "Compra",
    "Procurou Empresa": "Procurou empresa",
}


def limpar_nome_feature(nome):
    nome = str(nome).replace("cat__", "")

    for var in VARIAVEIS_ORIGINAIS:
        prefixo = var + "_"
        if nome.startswith(prefixo):
            categoria = nome[len(prefixo):]
            return var, categoria

    return "variavel_desconhecida", nome


def obter_nomes_features(pipeline):
    preprocessador = pipeline.named_steps["preprocessador"]

    try:
        return np.array(preprocessador.get_feature_names_out())
    except Exception:
        return np.array(preprocessador.get_feature_names())


def criar_label_grafico(df):
    variavel_curta = df["variavel"].map(VARIAVEIS_ABREV).fillna(df["variavel"])
    return variavel_curta.astype(str) + ": " + df["categoria"].astype(str)


def encurtar_label(texto, max_chars=70):
    texto = " ".join(str(texto).split())
    return shorten(texto, width=max_chars, placeholder="...")


def grafico_barras(
    df,
    coluna_valor,
    coluna_label,
    titulo,
    arquivo,
    top_n=10,
    max_chars=70,
    xlabel=None,
    casas_decimais=2
):
    """
    Gráfico pensado para relatório:
    - top 10 apenas;
    - labels encurtadas;
    - fonte maior;
    - imagem em tamanho razoável.
    """

    dados = df.head(top_n).copy()
    dados["label_plot"] = dados[coluna_label].apply(
        lambda x: encurtar_label(x, max_chars=max_chars)
    )

    dados = dados.sort_values(coluna_valor, ascending=True)

    altura = max(5.5, top_n * 0.55)
    fig, ax = plt.subplots(figsize=(11, altura))

    y = np.arange(len(dados))

    ax.barh(y, dados[coluna_valor])

    ax.set_yticks(y)
    ax.set_yticklabels(dados["label_plot"], fontsize=10)

    if xlabel is None:
        xlabel = coluna_valor

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_title(titulo, fontsize=13, pad=14)

    ax.tick_params(axis="x", labelsize=10)
    ax.grid(axis="x", alpha=0.25)

    maior_valor = dados[coluna_valor].max()
    margem = maior_valor * 0.12 if maior_valor != 0 else 0.1
    ax.set_xlim(0, maior_valor + margem)

    for i, valor in enumerate(dados[coluna_valor]):
        ax.text(
            valor + margem * 0.15,
            i,
            f"{valor:.{casas_decimais}f}",
            va="center",
            fontsize=9
        )

    fig.subplots_adjust(left=0.42, right=0.96, top=0.88, bottom=0.14)

    fig.savefig(
        PASTA_GRAFICOS / arquivo,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


def interpretar_regressao_logistica():
    print("\nInterpretando Regressão Logística...")

    pipeline = joblib.load(ARQ_REGRESSAO)

    nomes_features = obter_nomes_features(pipeline)
    modelo = pipeline.named_steps["modelo"]

    coeficientes = modelo.coef_[0]

    df = pd.DataFrame({
        "feature_codificada": nomes_features,
        "coeficiente": coeficientes
    })

    variaveis = df["feature_codificada"].apply(limpar_nome_feature)

    df["variavel"] = variaveis.apply(lambda x: x[0])
    df["categoria"] = variaveis.apply(lambda x: x[1])

    df["label_grafico"] = criar_label_grafico(df)
    df["abs_coeficiente"] = df["coeficiente"].abs()
    df["odds_ratio"] = np.exp(df["coeficiente"])

    positivos = (
        df.sort_values("coeficiente", ascending=False)
        .head(30)
        .copy()
    )

    negativos = (
        df.sort_values("coeficiente", ascending=True)
        .head(30)
        .copy()
    )

    positivos.to_csv(
        PASTA_INTERPRETACAO / "regressao_top_aumentam_probabilidade.csv",
        index=False
    )

    negativos.to_csv(
        PASTA_INTERPRETACAO / "regressao_top_reduzem_probabilidade.csv",
        index=False
    )

    agregada = (
        df.groupby("variavel", as_index=False)
        .agg(
            media_abs_coeficiente=("abs_coeficiente", "mean"),
            max_abs_coeficiente=("abs_coeficiente", "max"),
            quantidade_categorias=("categoria", "count")
        )
        .sort_values("media_abs_coeficiente", ascending=False)
    )

    agregada.to_csv(
        PASTA_INTERPRETACAO / "regressao_importancia_por_variavel.csv",
        index=False
    )

    grafico_barras(
        positivos,
        coluna_valor="coeficiente",
        coluna_label="label_grafico",
        titulo="Categorias que mais aumentam a probabilidade de resolução\nRegressão Logística",
        arquivo="regressao_top_aumentam_probabilidade.png",
        top_n=10,
        max_chars=68,
        xlabel="Coeficiente"
    )

    negativos_plot = negativos.copy()
    negativos_plot["coeficiente_abs_plot"] = negativos_plot["coeficiente"].abs()

    grafico_barras(
        negativos_plot,
        coluna_valor="coeficiente_abs_plot",
        coluna_label="label_grafico",
        titulo="Categorias que mais reduzem a probabilidade de resolução\nRegressão Logística",
        arquivo="regressao_top_reduzem_probabilidade.png",
        top_n=10,
        max_chars=68,
        xlabel="Valor absoluto do coeficiente"
    )

    grafico_barras(
        agregada,
        coluna_valor="media_abs_coeficiente",
        coluna_label="variavel",
        titulo="Importância média por variável\nRegressão Logística",
        arquivo="regressao_importancia_por_variavel.png",
        top_n=10,
        max_chars=45,
        xlabel="Média do valor absoluto dos coeficientes",
        casas_decimais=3
    )

    print("Arquivos da Regressão Logística salvos.")


def interpretar_xgboost():
    print("\nInterpretando XGBoost...")

    pipeline = joblib.load(ARQ_XGBOOST)

    nomes_features = obter_nomes_features(pipeline)
    modelo = pipeline.named_steps["modelo"]

    importancias = modelo.feature_importances_

    df = pd.DataFrame({
        "feature_codificada": nomes_features,
        "importancia": importancias
    })

    variaveis = df["feature_codificada"].apply(limpar_nome_feature)

    df["variavel"] = variaveis.apply(lambda x: x[0])
    df["categoria"] = variaveis.apply(lambda x: x[1])
    df["label_grafico"] = criar_label_grafico(df)

    df = df.sort_values("importancia", ascending=False)

    df.head(50).to_csv(
        PASTA_INTERPRETACAO / "xgboost_top_features.csv",
        index=False
    )

    agregada = (
        df.groupby("variavel", as_index=False)
        .agg(
            importancia_total=("importancia", "sum"),
            importancia_media=("importancia", "mean"),
            quantidade_categorias=("categoria", "count")
        )
        .sort_values("importancia_total", ascending=False)
    )

    agregada.to_csv(
        PASTA_INTERPRETACAO / "xgboost_importancia_por_variavel.csv",
        index=False
    )

    grafico_barras(
        df,
        coluna_valor="importancia",
        coluna_label="label_grafico",
        titulo="Categorias mais importantes\nXGBoost",
        arquivo="xgboost_top_features.png",
        top_n=10,
        max_chars=68,
        xlabel="Importância",
        casas_decimais=3
    )

    grafico_barras(
        agregada,
        coluna_valor="importancia_total",
        coluna_label="variavel",
        titulo="Importância agregada por variável\nXGBoost",
        arquivo="xgboost_importancia_por_variavel.png",
        top_n=10,
        max_chars=45,
        xlabel="Importância total",
        casas_decimais=3
    )

    print("Arquivos do XGBoost salvos.")


def main():
    if not ARQ_REGRESSAO.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ_REGRESSAO}")

    if not ARQ_XGBOOST.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {ARQ_XGBOOST}")

    interpretar_regressao_logistica()
    interpretar_xgboost()

    print("\nInterpretação concluída.")
    print("CSVs salvos em:", PASTA_INTERPRETACAO)
    print("Gráficos salvos em:", PASTA_GRAFICOS)

    print("\nArquivos PNG gerados:")
    for arquivo in sorted(PASTA_GRAFICOS.glob("*.png")):
        print("-", arquivo)


if __name__ == "__main__":
    main()
