from pathlib import Path
import duckdb

PASTA_PARTES = Path("data/processed/parts")
PADRAO_PARQUET = str(PASTA_PARTES / "*.parquet")

con = duckdb.connect()

print("Arquivos parquet encontrados:")
arquivos = sorted(PASTA_PARTES.glob("*.parquet"))

for arq in arquivos[:10]:
    print("-", arq.name)

print(f"\nTotal de arquivos parquet: {len(arquivos)}")

if not arquivos:
    raise RuntimeError("Nenhum arquivo parquet encontrado em data/processed/parts")

BASE = f"read_parquet('{PADRAO_PARQUET}', union_by_name=True)"

print("\nColunas da base:")
colunas = con.execute(f"""
    DESCRIBE SELECT * FROM {BASE}
""").fetchdf()

print(colunas)

print("\nQuantidade total de linhas:")
total = con.execute(f"""
    SELECT COUNT(*) AS total
    FROM {BASE}
""").fetchdf()

print(total)

print("\nDistribuição do target:")
target = con.execute(f"""
    SELECT
        target_resolvida,
        COUNT(*) AS quantidade,
        ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentual
    FROM {BASE}
    WHERE target_resolvida IS NOT NULL
    GROUP BY target_resolvida
    ORDER BY target_resolvida
""").fetchdf()

print(target)

print("\nDistribuição por ano de finalização:")
anos = con.execute(f"""
    SELECT
        ano_finalizacao,
        COUNT(*) AS quantidade,
        ROUND(AVG(target_resolvida), 4) AS taxa_resolucao
    FROM {BASE}
    WHERE ano_finalizacao IS NOT NULL
    GROUP BY ano_finalizacao
    ORDER BY ano_finalizacao
""").fetchdf()

print(anos)

print("\nTop 20 segmentos:")
segmentos = con.execute(f"""
    SELECT
        COALESCE("Segmento de Mercado", 'Não informado') AS segmento,
        COUNT(*) AS quantidade,
        ROUND(AVG(target_resolvida), 4) AS taxa_resolucao
    FROM {BASE}
    WHERE ano_finalizacao IS NOT NULL
    GROUP BY segmento
    ORDER BY quantidade DESC
    LIMIT 20
""").fetchdf()

print(segmentos)

print("\nTop 20 grupos de problema:")
grupos = con.execute(f"""
    SELECT
        COALESCE("Grupo Problema", 'Não informado') AS grupo_problema,
        COUNT(*) AS quantidade,
        ROUND(AVG(target_resolvida), 4) AS taxa_resolucao
    FROM {BASE}
    WHERE ano_finalizacao IS NOT NULL
    GROUP BY grupo_problema
    ORDER BY quantidade DESC
    LIMIT 20
""").fetchdf()

print(grupos)

Path("data/analysis").mkdir(parents=True, exist_ok=True)

anos.to_csv("data/analysis/resumo_por_ano.csv", index=False)
target.to_csv("data/analysis/resumo_target.csv", index=False)
segmentos.to_csv("data/analysis/resumo_segmentos.csv", index=False)
grupos.to_csv("data/analysis/resumo_grupos_problema.csv", index=False)

print("\nResumos salvos em data/analysis/")