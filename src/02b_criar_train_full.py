from pathlib import Path
import duckdb

PASTA_PARTES = Path("data/processed/parts")
PASTA_MODELAGEM = Path("data/modelagem")

PASTA_MODELAGEM.mkdir(parents=True, exist_ok=True)

PADRAO_PARQUET = str(PASTA_PARTES / "*.parquet")
BASE = f"read_parquet('{PADRAO_PARQUET}', union_by_name=True)"

con = duckdb.connect()

FEATURES = [
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

COLUNAS_SQL = ",\n            ".join(
    [
        f"COALESCE(CAST(\"{c}\" AS VARCHAR), 'Não informado') AS \"{c}\""
        for c in FEATURES
    ]
    + ["CAST(target_resolvida AS INTEGER) AS target_resolvida"]
)

saida = PASTA_MODELAGEM / "train_full.parquet"

query = f"""
    SELECT
        {COLUNAS_SQL}
    FROM {BASE}
    WHERE
        ano_finalizacao BETWEEN 2014 AND 2024
        AND ano_finalizacao IS NOT NULL
        AND target_resolvida IS NOT NULL
"""

print("Gerando treino completo 2014-2024...")
print("Saída:", saida)

con.execute(f"""
    COPY ({query})
    TO '{saida}'
    (FORMAT PARQUET)
""")

resumo = con.execute(f"""
    SELECT
        COUNT(*) AS linhas,
        SUM(target_resolvida) AS resolvidas,
        COUNT(*) - SUM(target_resolvida) AS nao_resolvidas,
        ROUND(AVG(target_resolvida), 4) AS taxa_resolucao
    FROM read_parquet('{saida}')
""").fetchdf()

print(resumo)
