from pathlib import Path
import duckdb

PASTA_PARTES = Path("data/processed/parts")
PASTA_MODELAGEM = Path("data/modelagem")

PASTA_MODELAGEM.mkdir(parents=True, exist_ok=True)

PADRAO_PARQUET = str(PASTA_PARTES / "*.parquet")

con = duckdb.connect()

# Importante: union_by_name=True porque alguns parquets têm colunas faltando
BASE = f"read_parquet('{PADRAO_PARQUET}', union_by_name=True)"

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

# Converte tudo para VARCHAR antes do COALESCE.
# Isso evita erro quando a coluna é numérica, como Ano Abertura e Mês Abertura.
COLUNAS_SQL = ",\n            ".join(
    [
        f"COALESCE(CAST(\"{c}\" AS VARCHAR), 'Não informado') AS \"{c}\""
        for c in FEATURES
    ]
    + ["CAST(target_resolvida AS INTEGER) AS target_resolvida"]
)


def salvar_conjunto(nome, condicao_sql, limite=None):
    saida = PASTA_MODELAGEM / f"{nome}.parquet"

    query = f"""
        SELECT
            {COLUNAS_SQL}
        FROM {BASE}
        WHERE
            {condicao_sql}
            AND ano_finalizacao IS NOT NULL
            AND target_resolvida IS NOT NULL
    """

    if limite is not None:
        query += f"""
        ORDER BY random()
        LIMIT {limite}
        """

    print(f"\nGerando {nome}...")
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


# Treino: amostra dos anos anteriores
salvar_conjunto(
    nome="train",
    condicao_sql="ano_finalizacao BETWEEN 2014 AND 2024",
    limite=500_000
)

# Validação: amostra de 2025
salvar_conjunto(
    nome="valid",
    condicao_sql="ano_finalizacao = 2025",
    limite=200_000
)

# Teste final: 2026 completo
salvar_conjunto(
    nome="test",
    condicao_sql="ano_finalizacao = 2026",
    limite=None
)

print("\nConjuntos criados em data/modelagem/")
