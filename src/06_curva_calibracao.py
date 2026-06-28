from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

PASTA_RESULTADOS = Path("data/resultados")
PASTA_GRAFICOS = PASTA_RESULTADOS / "graficos"
PASTA_GRAFICOS.mkdir(parents=True, exist_ok=True)

preds = pd.read_parquet(PASTA_RESULTADOS / "predicoes_teste_melhor_modelo.parquet")

y_true = preds["y_true"]
y_proba = preds["y_proba"]

prob_true, prob_pred = calibration_curve(
    y_true,
    y_proba,
    n_bins=10,
    strategy="quantile"
)

plt.figure(figsize=(6, 6))
plt.plot(prob_pred, prob_true, marker="o", label="Modelo")
plt.plot([0, 1], [0, 1], linestyle="--", label="Calibração perfeita")
plt.xlabel("Probabilidade média prevista")
plt.ylabel("Frequência real de resolução")
plt.title("Curva de calibração - modelo final")
plt.legend()
plt.tight_layout()
plt.savefig(PASTA_GRAFICOS / "curva_calibracao_modelo_final.png", dpi=200)
plt.close()

print("Curva de calibração salva em:")
print(PASTA_GRAFICOS / "curva_calibracao_modelo_final.png")
