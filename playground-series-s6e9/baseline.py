#!/usr/bin/env python3
"""S6E9 first baseline: HistGradientBoosting, ROC AUC."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" if (ROOT / "data" / "train.csv").exists() else ROOT
TARGET = "Will_Buy_EV"
ID_COL = "id"

NUMERIC = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CATEGORICAL = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]


def encode_target(s: pd.Series) -> np.ndarray:
    mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0, True: 1, False: 0}
    y = s.map(mapping)
    if y.isna().any():
        leftover = s[y.isna()].unique()[:10]
        raise ValueError(f"unmapped target values: {leftover}")
    return y.astype(int).to_numpy()


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = encode_target(train[TARGET])
    X = train[NUMERIC + CATEGORICAL]
    X_test = test[NUMERIC + CATEGORICAL]

    print("train", train.shape, "test", test.shape, "pos_rate", float(y.mean()))
    print("missing train", int(X.isna().sum().sum()), "test", int(X_test.isna().sum().sum()))

    model = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", "passthrough", NUMERIC),
                        (
                            "cat",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                                encoded_missing_value=-1,
                            ),
                            CATEGORICAL,
                        ),
                    ]
                ),
            ),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_depth=6,
                    max_iter=250,
                    l2_regularization=0.1,
                    min_samples_leaf=40,
                    categorical_features=list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL))),
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=20,
                    random_state=42,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    fold_scores = []

    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        model.fit(X.iloc[tr], y[tr])
        p_va = model.predict_proba(X.iloc[va])[:, 1]
        p_te = model.predict_proba(X_test)[:, 1]
        oof[va] = p_va
        test_pred += p_te / cv.n_splits
        auc = roc_auc_score(y[va], p_va)
        fold_scores.append(auc)
        print(f"fold {fold} auc={auc:.5f}")

    cv_auc = roc_auc_score(y, oof)
    print(f"cv mean={float(np.mean(fold_scores)):.5f} oof={cv_auc:.5f}")

    out = ROOT / "submission.csv"
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred}).to_csv(out, index=False)
    print("wrote", out, "rows", len(test_pred), "mean_pred", float(test_pred.mean()))


if __name__ == "__main__":
    main()
