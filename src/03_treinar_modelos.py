from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
    confusion_matrix,
)

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =========================
# CONFIGURAÇÕES
# =========================

PASTA_MODELAGEM = Path("data/modelagem")
PASTA_RESULTADOS = Path("data/resultados")
PASTA_MODELOS = Path("models")

PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
PASTA_MODELOS.mkdir(parents=True, exist_ok=True)

ARQ_TRAIN = PASTA_MODELAGEM / "train.parquet"
ARQ_VALID = PASTA_MODELAGEM / "valid.parquet"
ARQ_TEST = PASTA_MODELAGEM / "test.parquet"

TARGET = "target_resolvida"
RANDOM_STATE = 42

# Amostras para controlar tempo de execução.
# Depois, se rodar tranquilo, podemos aumentar.
TRAIN_SAMPLE_LR = 500_000
TRAIN_SAMPLE_TREE = 500_000
TRAIN_SAMPLE_XGB = 500_000
TRAIN_SAMPLE_SVM = 100_000


# =========================
# FUNÇÕES AUXILIARES
# =========================

def carregar_dados():
    print("Lendo conjuntos de modelagem...")

    train = pd.read_parquet(ARQ_TRAIN)
    valid = pd.read_parquet(ARQ_VALID)
    test = pd.read_parquet(ARQ_TEST)

    print("Train:", train.shape)
    print("Valid:", valid.shape)
    print("Test:", test.shape)

    return train, valid, test


def amostrar(df, n):
    if n is None or len(df) <= n:
        return df

    return df.sample(n=n, random_state=RANDOM_STATE)


def separar_xy(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    return X, y


def criar_preprocessador(features):
    try:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )
    except TypeError:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=True
        )

    preprocessador = ColumnTransformer(
        transformers=[
            ("cat", encoder, features)
        ],
        remainder="drop"
    )

    return preprocessador


def obter_proba(modelo, X):
    return modelo.predict_proba(X)[:, 1]


def melhor_threshold(y_true, y_proba):
    thresholds = np.arange(0.10, 0.91, 0.01)

    melhor_t = 0.50
    melhor_f1 = -1

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_t = t

    return melhor_t, melhor_f1


def avaliar_modelo(nome, y_true, y_proba, threshold, conjunto):
    y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "conjunto": conjunto,
        "modelo": nome,
        "threshold": threshold,

        "accuracy": accuracy_score(y_true, y_pred),

        "precision_resolvida": precision_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "recall_resolvida": recall_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),
        "f1_resolvida": f1_score(
            y_true, y_pred, pos_label=1, zero_division=0
        ),

        "precision_macro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),

        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_proba),

        "tn_nao_resolvida": tn,
        "fp_nao_resolvida_classificada_como_resolvida": fp,
        "fn_resolvida_classificada_como_nao_resolvida": fn,
        "tp_resolvida": tp,
    }


# =========================
# BASELINES
# =========================

def baseline_media_geral(train, valid, test):
    nome = "Baseline - média geral"

    taxa = train[TARGET].mean()

    y_valid = valid[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    proba_valid = np.full(len(valid), taxa)
    proba_test = np.full(len(test), taxa)

    threshold, f1_valid = melhor_threshold(y_valid, proba_valid)

    print("\nBaseline média geral")
    print("Taxa do treino:", round(taxa, 4))
    print("Melhor threshold validação:", round(threshold, 2))
    print("F1 macro validação:", round(f1_valid, 4))

    res_valid = avaliar_modelo(
        nome, y_valid, proba_valid, threshold, "validacao"
    )

    res_test = avaliar_modelo(
        nome, y_test, proba_test, threshold, "teste"
    )

    return res_valid, res_test


def baseline_por_grupo(train, valid, test):
    nome = "Baseline - média por Grupo Problema"
    coluna = "Grupo Problema"

    taxa_global = train[TARGET].mean()
    taxas_por_grupo = train.groupby(coluna)[TARGET].mean().to_dict()

    y_valid = valid[TARGET].astype(int)
    y_test = test[TARGET].astype(int)

    proba_valid = (
        valid[coluna]
        .map(taxas_por_grupo)
        .fillna(taxa_global)
        .to_numpy()
    )

    proba_test = (
        test[coluna]
        .map(taxas_por_grupo)
        .fillna(taxa_global)
        .to_numpy()
    )

    threshold, f1_valid = melhor_threshold(y_valid, proba_valid)

    print("\nBaseline média por Grupo Problema")
    print("Melhor threshold validação:", round(threshold, 2))
    print("F1 macro validação:", round(f1_valid, 4))

    res_valid = avaliar_modelo(
        nome, y_valid, proba_valid, threshold, "validacao"
    )

    res_test = avaliar_modelo(
        nome, y_test, proba_test, threshold, "teste"
    )

    return res_valid, res_test


# =========================
# TREINO E AVALIAÇÃO
# =========================

def treinar_e_avaliar(nome, modelo, train, valid, test, features, n_amostra):
    print("\n==============================")
    print("Treinando:", nome)

    train_usado = amostrar(train, n_amostra)

    print("Amostra de treino usada:", train_usado.shape)

    X_train, y_train = separar_xy(train_usado)
    X_valid, y_valid = separar_xy(valid)
    X_test, y_test = separar_xy(test)

    preprocessador = criar_preprocessador(features)

    pipeline = Pipeline([
        ("preprocessador", preprocessador),
        ("modelo", modelo)
    ])

    pipeline.fit(X_train, y_train)

    print("Prevendo validação...")
    proba_valid = obter_proba(pipeline, X_valid)

    threshold, f1_valid = melhor_threshold(y_valid, proba_valid)

    print("Melhor threshold validação:", round(threshold, 2))
    print("F1 macro validação:", round(f1_valid, 4))

    print("Prevendo teste...")
    proba_test = obter_proba(pipeline, X_test)

    res_valid = avaliar_modelo(
        nome, y_valid, proba_valid, threshold, "validacao"
    )

    res_test = avaliar_modelo(
        nome, y_test, proba_test, threshold, "teste"
    )

    predicoes_test = pd.DataFrame({
        "modelo": nome,
        "y_true": y_test.to_numpy(),
        "y_proba": proba_test,
        "y_pred": (proba_test >= threshold).astype(int),
    })

    return pipeline, res_valid, res_test, predicoes_test


# =========================
# MAIN
# =========================

def main():
    train, valid, test = carregar_dados()

    features = [c for c in train.columns if c != TARGET]

    print("\nFeatures usadas:")
    for f in features:
        print("-", f)

    resultados_valid = []
    resultados_test = []
    modelos_treinados = {}
    predicoes_teste = {}

    # -------------------------
    # Baselines
    # -------------------------

    print("\n==============================")
    print("Calculando baselines...")

    res_valid, res_test = baseline_media_geral(train, valid, test)
    resultados_valid.append(res_valid)
    resultados_test.append(res_test)

    res_valid, res_test = baseline_por_grupo(train, valid, test)
    resultados_valid.append(res_valid)
    resultados_test.append(res_test)

    # -------------------------
    # 1. Regressão Logística
    # Família: Linear / Regressores
    # -------------------------

    modelo_lr = LogisticRegression(
        C=0.1,
        max_iter=400,
        solver="saga",
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    pipe, res_valid, res_test, preds = treinar_e_avaliar(
        nome="Regressão Logística",
        modelo=modelo_lr,
        train=train,
        valid=valid,
        test=test,
        features=features,
        n_amostra=TRAIN_SAMPLE_LR
    )

    modelos_treinados["Regressão Logística"] = pipe
    predicoes_teste["Regressão Logística"] = preds
    resultados_valid.append(res_valid)
    resultados_test.append(res_test)

    # -------------------------
    # 2. Árvore de Decisão
    # Família: Árvores
    # -------------------------

    modelo_arvore = DecisionTreeClassifier(
        max_depth=20,
        min_samples_leaf=500,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    pipe, res_valid, res_test, preds = treinar_e_avaliar(
        nome="Árvore de Decisão",
        modelo=modelo_arvore,
        train=train,
        valid=valid,
        test=test,
        features=features,
        n_amostra=TRAIN_SAMPLE_TREE
    )

    modelos_treinados["Árvore de Decisão"] = pipe
    predicoes_teste["Árvore de Decisão"] = preds
    resultados_valid.append(res_valid)
    resultados_test.append(res_test)

    # -------------------------
    # 3. XGBoost
    # Família: Ensemble / Boosting
    # -------------------------

    modelo_xgb = XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    pipe, res_valid, res_test, preds = treinar_e_avaliar(
        nome="XGBoost",
        modelo=modelo_xgb,
        train=train,
        valid=valid,
        test=test,
        features=features,
        n_amostra=TRAIN_SAMPLE_XGB
    )

    modelos_treinados["XGBoost"] = pipe
    predicoes_teste["XGBoost"] = preds
    resultados_valid.append(res_valid)
    resultados_test.append(res_test)

    # -------------------------
    # 4. SVM Linear calibrado
    # Família: SVM
    # -------------------------

    modelo_svm = CalibratedClassifierCV(
        estimator=LinearSVC(
            C=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=3000
        ),
        method="sigmoid",
        cv=3
    )

    pipe, res_valid, res_test, preds = treinar_e_avaliar(
        nome="SVM Linear calibrado",
        modelo=modelo_svm,
        train=train,
        valid=valid,
        test=test,
        features=features,
        n_amostra=TRAIN_SAMPLE_SVM
    )

    modelos_treinados["SVM Linear calibrado"] = pipe
    predicoes_teste["SVM Linear calibrado"] = preds
    resultados_valid.append(res_valid)
    resultados_test.append(res_test)

    # -------------------------
    # Salvar resultados
    # -------------------------

    df_valid = pd.DataFrame(resultados_valid).sort_values(
        by="f1_macro",
        ascending=False
    )

    df_test = pd.DataFrame(resultados_test).sort_values(
        by="f1_macro",
        ascending=False
    )

    arq_valid = PASTA_RESULTADOS / "resultados_validacao.csv"
    arq_test = PASTA_RESULTADOS / "resultados_teste.csv"

    df_valid.to_csv(arq_valid, index=False)
    df_test.to_csv(arq_test, index=False)

    print("\n==============================")
    print("Resultados na validação:")
    print(df_valid[[
        "modelo",
        "threshold",
        "accuracy",
        "f1_macro",
        "roc_auc",
        "brier_score",
        "log_loss"
    ]])

    print("\n==============================")
    print("Resultados no teste:")
    print(df_test[[
        "modelo",
        "threshold",
        "accuracy",
        "f1_macro",
        "roc_auc",
        "brier_score",
        "log_loss"
    ]])

    print("\nArquivos salvos:")
    print("-", arq_valid)
    print("-", arq_test)

    # -------------------------
    # Escolher melhor modelo
    # -------------------------

    candidatos = df_valid[
        ~df_valid["modelo"].str.startswith("Baseline")
    ].copy()

    melhor_classificacao = candidatos.sort_values(
        by="f1_macro",
        ascending=False
    ).iloc[0]["modelo"]

    melhor_probabilidade = candidatos.sort_values(
        by=["brier_score", "log_loss"],
        ascending=[True, True]
    ).iloc[0]["modelo"]

    print("\nMelhor modelo para classificação:", melhor_classificacao)
    print("Melhor modelo para probabilidade:", melhor_probabilidade)

    # Como o painel é probabilístico, salvamos como modelo final o melhor em probabilidade.
    melhor_nome = melhor_probabilidade
    melhor_modelo = modelos_treinados[melhor_nome]

    arq_modelo = PASTA_MODELOS / "modelo_final.joblib"
    arq_features = PASTA_MODELOS / "features.joblib"
    arq_predicoes = PASTA_RESULTADOS / "predicoes_teste_melhor_modelo.parquet"

    joblib.dump(melhor_modelo, arq_modelo)
    joblib.dump(features, arq_features)

    predicoes_teste[melhor_nome].to_parquet(arq_predicoes, index=False)

    print("Modelo final salvo em:", arq_modelo)
    print("Features salvas em:", arq_features)
    print("Predições do teste salvas em:", arq_predicoes)
    
    def nome_arquivo_modelo(nome):
        nome = nome.lower()
        nome = nome.replace(" ", "_")
        nome = nome.replace("ç", "c")
        nome = nome.replace("ã", "a")
        nome = nome.replace("á", "a")
        nome = nome.replace("é", "e")
        nome = nome.replace("í", "i")
        nome = nome.replace("ó", "o")
        nome = nome.replace("ú", "u")
        nome = nome.replace("ã", "a")
        return nome

    for nome, modelo in modelos_treinados.items():
        caminho = PASTA_MODELOS / f"{nome_arquivo_modelo(nome)}.joblib"
        joblib.dump(modelo, caminho)
        print("Modelo salvo:", caminho)


if __name__ == "__main__":
    main()
