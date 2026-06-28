from pathlib import Path
import warnings

import numpy as np
import pandas as pd

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


PASTA_MODELAGEM = Path("data/modelagem")
PASTA_RESULTADOS = Path("data/resultados")
PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

ARQ_TRAIN = PASTA_MODELAGEM / "train.parquet"
ARQ_VALID = PASTA_MODELAGEM / "valid.parquet"

TARGET = "target_resolvida"
RANDOM_STATE = 42

TRAIN_SAMPLE_LR = 500_000
TRAIN_SAMPLE_TREE = 500_000
TRAIN_SAMPLE_XGB = 500_000
TRAIN_SAMPLE_SVM = 100_000


def carregar_dados():
    print("Lendo dados...")

    train = pd.read_parquet(ARQ_TRAIN)
    valid = pd.read_parquet(ARQ_VALID)

    print("Train:", train.shape)
    print("Valid:", valid.shape)

    return train, valid


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

    return ColumnTransformer(
        transformers=[
            ("cat", encoder, features)
        ],
        remainder="drop"
    )


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


def avaliar(nome, familia, config_nome, params, y_true, y_proba, threshold):
    y_proba = np.clip(y_proba, 1e-15, 1 - 1e-15)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    return {
        "modelo": nome,
        "familia": familia,
        "configuracao": config_nome,
        "params": str(params),
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_resolvida": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_resolvida": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_resolvida": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba, labels=[0, 1]),
        "brier_score": brier_score_loss(y_true, y_proba),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def treinar_configuracao(nome, familia, config_nome, modelo, params, train, valid, features, n_amostra):
    print("\n==============================")
    print("Modelo:", nome)
    print("Configuração:", config_nome)
    print("Parâmetros:", params)

    train_usado = amostrar(train, n_amostra)

    X_train, y_train = separar_xy(train_usado)
    X_valid, y_valid = separar_xy(valid)

    pipeline = Pipeline([
        ("preprocessador", criar_preprocessador(features)),
        ("modelo", modelo)
    ])

    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_valid)[:, 1]

    threshold, f1_valid = melhor_threshold(y_valid, y_proba)

    print("Threshold:", round(threshold, 2))
    print("F1 macro:", round(f1_valid, 4))

    resultado = avaliar(
        nome=nome,
        familia=familia,
        config_nome=config_nome,
        params=params,
        y_true=y_valid,
        y_proba=y_proba,
        threshold=threshold
    )

    return resultado


def main():
    train, valid = carregar_dados()

    features = [c for c in train.columns if c != TARGET]

    resultados = []

    # =========================
    # REGRESSÃO LOGÍSTICA
    # =========================

    configs_lr = [
        ("LR_C_0.1", {"C": 0.1}),
        ("LR_C_1.0", {"C": 1.0}),
        ("LR_C_10.0", {"C": 10.0}),
    ]

    for config_nome, params in configs_lr:
        modelo = LogisticRegression(
            C=params["C"],
            max_iter=500,
            solver="saga",
            n_jobs=-1,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

        resultado = treinar_configuracao(
            nome="Regressão Logística",
            familia="Linear / Regressores",
            config_nome=config_nome,
            modelo=modelo,
            params=params,
            train=train,
            valid=valid,
            features=features,
            n_amostra=TRAIN_SAMPLE_LR
        )

        resultados.append(resultado)

    # =========================
    # ÁRVORE DE DECISÃO
    # =========================

    configs_tree = [
        ("TREE_simples", {"max_depth": 10, "min_samples_leaf": 1000}),
        ("TREE_media", {"max_depth": 20, "min_samples_leaf": 500}),
        ("TREE_complexa", {"max_depth": 30, "min_samples_leaf": 200}),
    ]

    for config_nome, params in configs_tree:
        modelo = DecisionTreeClassifier(
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            random_state=RANDOM_STATE
        )

        resultado = treinar_configuracao(
            nome="Árvore de Decisão",
            familia="Árvores",
            config_nome=config_nome,
            modelo=modelo,
            params=params,
            train=train,
            valid=valid,
            features=features,
            n_amostra=TRAIN_SAMPLE_TREE
        )

        resultados.append(resultado)

    # =========================
    # XGBOOST
    # =========================

    configs_xgb = [
        (
            "XGB_simples",
            {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.10,
                "min_child_weight": 5,
            }
        ),
        (
            "XGB_medio",
            {
                "n_estimators": 250,
                "max_depth": 6,
                "learning_rate": 0.08,
                "min_child_weight": 3,
            }
        ),
        (
            "XGB_robusto",
            {
                "n_estimators": 400,
                "max_depth": 4,
                "learning_rate": 0.05,
                "min_child_weight": 5,
            }
        ),
    ]

    for config_nome, params in configs_xgb:
        modelo = XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            min_child_weight=params["min_child_weight"],
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )

        resultado = treinar_configuracao(
            nome="XGBoost",
            familia="Ensemble / Boosting",
            config_nome=config_nome,
            modelo=modelo,
            params=params,
            train=train,
            valid=valid,
            features=features,
            n_amostra=TRAIN_SAMPLE_XGB
        )

        resultados.append(resultado)

    # =========================
    # SVM LINEAR CALIBRADO
    # =========================

    configs_svm = [
        ("SVM_C_0.1", {"C": 0.1}),
        ("SVM_C_1.0", {"C": 1.0}),
        ("SVM_C_10.0", {"C": 10.0}),
    ]

    for config_nome, params in configs_svm:
        modelo_base = LinearSVC(
            C=params["C"],
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_iter=3000
        )

        modelo = CalibratedClassifierCV(
            estimator=modelo_base,
            method="sigmoid",
            cv=3
        )

        resultado = treinar_configuracao(
            nome="SVM Linear calibrado",
            familia="SVM",
            config_nome=config_nome,
            modelo=modelo,
            params=params,
            train=train,
            valid=valid,
            features=features,
            n_amostra=TRAIN_SAMPLE_SVM
        )

        resultados.append(resultado)

    df = pd.DataFrame(resultados)

    df = df.sort_values(
        by="f1_macro",
        ascending=False
    )

    saida = PASTA_RESULTADOS / "tuning_validacao.csv"
    df.to_csv(saida, index=False)

    print("\n==============================")
    print("RESULTADOS DO TUNING - ordenado por F1 macro")
    print(df[[
        "modelo",
        "configuracao",
        "threshold",
        "f1_macro",
        "roc_auc",
        "brier_score",
        "log_loss"
    ]])

    melhores_classificacao = (
        df.sort_values("f1_macro", ascending=False)
        .groupby("modelo")
        .head(1)
    )

    melhores_probabilidade = (
        df.sort_values(["brier_score", "log_loss"], ascending=[True, True])
        .groupby("modelo")
        .head(1)
    )

    melhores_classificacao.to_csv(
        PASTA_RESULTADOS / "melhores_configs_classificacao.csv",
        index=False
    )

    melhores_probabilidade.to_csv(
        PASTA_RESULTADOS / "melhores_configs_probabilidade.csv",
        index=False
    )

    print("\nMelhores configurações por F1 macro:")
    print(melhores_classificacao[[
        "modelo",
        "configuracao",
        "f1_macro",
        "roc_auc",
        "brier_score",
        "log_loss"
    ]])

    print("\nMelhores configurações por probabilidade:")
    print(melhores_probabilidade[[
        "modelo",
        "configuracao",
        "f1_macro",
        "roc_auc",
        "brier_score",
        "log_loss"
    ]])


if __name__ == "__main__":
    main()
