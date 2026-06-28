import pandas as pd
from pathlib import Path
import unicodedata
import re


# =========================
# CONFIGURAÇÕES
# =========================

PASTA_RAW = Path("data/raw")
PASTA_PROCESSADA = Path("data/processed")
PASTA_PARTES = PASTA_PROCESSADA / "parts"

PASTA_PROCESSADA.mkdir(parents=True, exist_ok=True)
PASTA_PARTES.mkdir(parents=True, exist_ok=True)

# Se quiser tentar gerar um arquivo único no final, mude para True.
# Para evitar estouro de memória, recomendo deixar False por enquanto.
SALVAR_BASE_UNICA = False


# =========================
# FUNÇÕES AUXILIARES
# =========================

def normalizar_nome(texto):
    """
    Normaliza nomes de colunas:
    - remove acentos
    - remove espaços
    - remove caracteres especiais
    - coloca em minúsculo
    """
    texto = str(texto).replace("\ufeff", "").strip()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-zA-Z0-9]", "", texto)
    return texto.lower()


def normalizar_valor(valor):
    """
    Normaliza valores textuais:
    Exemplo:
    'Não Resolvida' -> 'naoresolvida'
    'Finalizada avaliada' -> 'finalizadaavaliada'
    """
    if pd.isna(valor):
        return ""

    texto = str(valor).strip()
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = re.sub(r"[^a-zA-Z0-9]", "", texto)
    return texto.lower()


def parse_data_mista(serie):
    """
    Trata datas em dois formatos comuns:
    - dd/mm/aaaa
    - aaaa-mm-dd
    """
    s = serie.astype("string").str.strip()

    datas = pd.Series(pd.NaT, index=serie.index, dtype="datetime64[ns]")

    mask_iso = s.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    mask_br = s.str.match(r"^\d{2}/\d{2}/\d{4}$", na=False)

    datas.loc[mask_iso] = pd.to_datetime(
        s.loc[mask_iso],
        format="%Y-%m-%d",
        errors="coerce"
    )

    datas.loc[mask_br] = pd.to_datetime(
        s.loc[mask_br],
        format="%d/%m/%Y",
        errors="coerce"
    )

    # Fallback para algum formato inesperado
    mask_restante = ~(mask_iso | mask_br)
    if mask_restante.any():
        datas.loc[mask_restante] = pd.to_datetime(
            s.loc[mask_restante],
            errors="coerce",
            dayfirst=True
        )

    return datas


# =========================
# MAPA DE COLUNAS
# =========================

MAPA_COLUNAS = {
    "gestor": "Gestor",
    "canaldeorigem": "Canal de Origem",
    "regiao": "Região",
    "uf": "UF",
    "cidade": "Cidade",
    "sexo": "Sexo",
    "faixaetaria": "Faixa Etária",
    "anoabertura": "Ano Abertura",
    "mesabertura": "Mês Abertura",
    "dataabertura": "Data Abertura",
    "dataresposta": "Data Resposta",
    "dataanalise": "Data Análise",
    "datarecusa": "Data Recusa",
    "datafinalizacao": "Data Finalização",
    "prazoresposta": "Prazo Resposta",
    "prazoanalisegestor": "Prazo Analise Gestor",
    "temporesposta": "Tempo Resposta",
    "nomefantasia": "Nome Fantasia",
    "segmentodemercado": "Segmento de Mercado",
    "segmentomercado": "Segmento de Mercado",
    "area": "Área",
    "assunto": "Assunto",
    "grupoproblema": "Grupo Problema",
    "problema": "Problema",
    "comocomproucontratou": "Como Comprou Contratou",
    "procurouempresa": "Procurou Empresa",
    "respondida": "Respondida",
    "situacao": "Situação",
    "avaliacaoreclamacao": "Avaliação Reclamação",
    "notadoconsumidor": "Nota do Consumidor",
    "analisedarecusa": "Análise da Recusa",
}


COLUNAS_USADAS = [
    "Canal de Origem",
    "Região",
    "UF",
    "Cidade",
    "Sexo",
    "Faixa Etária",
    "Ano Abertura",
    "Mês Abertura",
    "Data Abertura",
    "Data Finalização",
    "Segmento de Mercado",
    "Área",
    "Assunto",
    "Grupo Problema",
    "Problema",
    "Como Comprou Contratou",
    "Procurou Empresa",
    "Situação",
    "Avaliação Reclamação",
]

COLUNAS_OBRIGATORIAS = [
    "Situação",
    "Avaliação Reclamação",
    "Data Finalização",
]


# =========================
# LEITURA ROBUSTA DOS CSVs
# =========================

def escolher_leitura(arquivo):
    """
    Tenta descobrir automaticamente:
    - encoding
    - separador
    - se precisa pular primeira linha

    Isso resolve casos em que alguns arquivos vêm com ; e outros com tabulação.
    """
    encodings = ["latin1", "cp1252", "utf-8-sig"]
    separadores = [";", "\t", ","]
    skiprows_opcoes = [0, 1]

    tentativas = []

    for enc in encodings:
        for sep in separadores:
            for skiprows in skiprows_opcoes:
                try:
                    df0 = pd.read_csv(
                        arquivo,
                        sep=sep,
                        encoding=enc,
                        nrows=0,
                        skiprows=skiprows
                    )

                    colunas = df0.columns.tolist()
                    colunas_norm = [normalizar_nome(c) for c in colunas]

                    acertos = sum(
                        1 for c in colunas_norm
                        if c in MAPA_COLUNAS
                    )

                    tentativas.append({
                        "acertos": acertos,
                        "encoding": enc,
                        "sep": sep,
                        "skiprows": skiprows,
                        "colunas": colunas
                    })

                except Exception:
                    continue

    if not tentativas:
        raise ValueError(f"Não consegui ler nem o cabeçalho do arquivo: {arquivo.name}")

    tentativas = sorted(
        tentativas,
        key=lambda x: x["acertos"],
        reverse=True
    )

    melhor = tentativas[0]

    if melhor["acertos"] < 5:
        print("\nNão encontrei colunas suficientes no arquivo:", arquivo.name)
        print("Melhor tentativa:")
        print("Encoding:", melhor["encoding"])
        print("Separador:", repr(melhor["sep"]))
        print("Skiprows:", melhor["skiprows"])
        print("Acertos:", melhor["acertos"])
        print("Colunas encontradas:")
        print(melhor["colunas"])

        print("\nPrimeiros bytes do arquivo:")
        with open(arquivo, "rb") as f:
            print(f.read(300))

        raise ValueError(f"Arquivo com cabeçalho inesperado: {arquivo.name}")

    print(
        f"Leitura escolhida: encoding={melhor['encoding']}, "
        f"sep={repr(melhor['sep'])}, skiprows={melhor['skiprows']}"
    )

    return melhor


def ler_csv_consumidor(arquivo):
    config = escolher_leitura(arquivo)

    colunas_originais = config["colunas"]

    colunas_para_ler = [
        c for c in colunas_originais
        if normalizar_nome(c) in MAPA_COLUNAS
    ]

    df = pd.read_csv(
        arquivo,
        sep=config["sep"],
        encoding=config["encoding"],
        skiprows=config["skiprows"],
        usecols=colunas_para_ler,
        low_memory=False
    )

    df = df.rename(columns={
        c: MAPA_COLUNAS[normalizar_nome(c)]
        for c in df.columns
        if normalizar_nome(c) in MAPA_COLUNAS
    })

    return df, config


# =========================
# PROCESSAMENTO DOS ARQUIVOS
# =========================

def processar_arquivo(arquivo):
    print("\n====================================")
    print("Processando:", arquivo.name)

    df, config = ler_csv_consumidor(arquivo)

    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]

    if faltando:
        print("Arquivo ignorado por falta de colunas obrigatórias:", arquivo.name)
        print("Colunas faltando:", faltando)
        print("Colunas encontradas:", df.columns.tolist())

        return {
            "arquivo": arquivo.name,
            "status": "ignorado_colunas_faltando",
            "linhas_lidas": len(df),
            "linhas_filtradas": 0,
            "encoding": config["encoding"],
            "sep": repr(config["sep"]),
            "skiprows": config["skiprows"],
        }

    linhas_lidas = len(df)

    situacao_norm = df["Situação"].map(normalizar_valor)
    avaliacao_norm = df["Avaliação Reclamação"].map(normalizar_valor)

    df = df[
        (situacao_norm == "finalizadaavaliada") &
        (avaliacao_norm.isin(["resolvida", "naoresolvida"]))
    ].copy()

    if df.empty:
        print("Nenhuma reclamação finalizada avaliada neste arquivo.")

        return {
            "arquivo": arquivo.name,
            "status": "sem_dados_filtrados",
            "linhas_lidas": linhas_lidas,
            "linhas_filtradas": 0,
            "encoding": config["encoding"],
            "sep": repr(config["sep"]),
            "skiprows": config["skiprows"],
        }

    # Recalcula após o filtro
    avaliacao_norm = df["Avaliação Reclamação"].map(normalizar_valor)

    df["target_resolvida"] = (avaliacao_norm == "resolvida").astype(int)

    df["Data Finalização"] = parse_data_mista(df["Data Finalização"])

    df["ano_finalizacao"] = df["Data Finalização"].dt.year
    df["mes_finalizacao"] = df["Data Finalização"].dt.month

    # Mantém apenas colunas úteis que existirem
    colunas_saida = [
        c for c in COLUNAS_USADAS
        if c in df.columns
    ] + [
        "ano_finalizacao",
        "mes_finalizacao",
        "target_resolvida"
    ]

    df = df[colunas_saida]

    # Padroniza valores vazios em colunas textuais
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].fillna("Não informado")

    arquivo_saida = PASTA_PARTES / f"{arquivo.stem}.parquet"
    df.to_parquet(arquivo_saida, index=False)

    print("Linhas lidas:", linhas_lidas)
    print("Linhas após filtro:", len(df))
    print("Resolvidas / Não resolvidas:")
    print(df["target_resolvida"].value_counts().sort_index())
    print("Salvo em:", arquivo_saida)

    return {
        "arquivo": arquivo.name,
        "status": "ok",
        "linhas_lidas": linhas_lidas,
        "linhas_filtradas": len(df),
        "resolvidas": int(df["target_resolvida"].sum()),
        "nao_resolvidas": int((df["target_resolvida"] == 0).sum()),
        "ano_min": int(df["ano_finalizacao"].min()) if df["ano_finalizacao"].notna().any() else None,
        "ano_max": int(df["ano_finalizacao"].max()) if df["ano_finalizacao"].notna().any() else None,
        "encoding": config["encoding"],
        "sep": repr(config["sep"]),
        "skiprows": config["skiprows"],
        "saida": str(arquivo_saida),
    }


def main():
    arquivos = sorted(PASTA_RAW.glob("*.csv"))

    if not arquivos:
        raise RuntimeError(f"Nenhum arquivo .csv encontrado em {PASTA_RAW}")

    logs = []

    for arquivo in arquivos:
        try:
            info = processar_arquivo(arquivo)
            logs.append(info)

        except Exception as e:
            print("\nERRO ao processar:", arquivo.name)
            print(type(e).__name__, e)

            logs.append({
                "arquivo": arquivo.name,
                "status": "erro",
                "erro": f"{type(e).__name__}: {e}",
            })

    catalogo = pd.DataFrame(logs)
    catalogo_saida = PASTA_PROCESSADA / "catalogo_processamento.csv"
    catalogo.to_csv(catalogo_saida, index=False)

    print("\n====================================")
    print("PROCESSAMENTO FINALIZADO")
    print("Catálogo salvo em:", catalogo_saida)
    print("\nResumo por status:")
    print(catalogo["status"].value_counts(dropna=False))

    if SALVAR_BASE_UNICA:
        print("\nGerando base única. Isso pode consumir bastante memória...")

        partes = []

        for parquet in sorted(PASTA_PARTES.glob("*.parquet")):
            partes.append(pd.read_parquet(parquet))

        base = pd.concat(partes, ignore_index=True)
        saida_unica = PASTA_PROCESSADA / "base_limpa.parquet"
        base.to_parquet(saida_unica, index=False)

        print("Base única salva em:", saida_unica)
        print("Shape:", base.shape)
        print(base["target_resolvida"].value_counts().sort_index())


if __name__ == "__main__":
    main()
