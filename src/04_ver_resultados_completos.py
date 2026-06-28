import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)

valid = pd.read_csv("data/resultados/resultados_validacao.csv")
test = pd.read_csv("data/resultados/resultados_teste.csv")

print("\nRESULTADOS - VALIDAÇÃO")
print(valid)

print("\nRESULTADOS - TESTE")
print(test)
