#!/usr/bin/env python3
"""S6E9 exp: swap HistGB -> LightGBM. Same folds, same features."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
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
    print("model LightGBM n_estimators=250 lr=0.08 max_depth=6")

    prep = ColumnTransformer(
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
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    fold_scores = []
    cat_idx = list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL)))

    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        X_tr = prep.fit_transform(X.iloc[tr])
        X_va = prep.transform(X.iloc[va])
        X_te = prep.transform(X_test)
        clf = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.08,
            max_depth=6,
            min_child_samples=40,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(
            X_tr,
            y[tr],
            eval_set=[(X_va, y[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        p_va = clf.predict_proba(X_va)[:, 1]
        p_te = clf.predict_proba(X_te)[:, 1]
        oof[va] = p_va
        test_pred += p_te / cv.n_splits
        auc = roc_auc_score(y[va], p_va)
        fold_scores.append(auc)
        print(f"fold {fold} auc={auc:.5f} best_iter={clf.best_iteration_}")

    cv_mean = float(np.mean(fold_scores))
    cv_auc = roc_auc_score(y, oof)
    print(f"cv mean={cv_mean:.5f} oof={cv_auc:.5f}")
    print("baseline histgb oof=0.94154")
    delta = cv_auc - 0.94154
    print(f"delta_oof={delta:+.5f} decision={'KEEP' if cv_auc > 0.94154 else 'DISCARD'}")

    out = DATA / "submission_lgbm.csv"
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred}).to_csv(out, index=False)
    print("wrote", out, "rows", len(test_pred), "mean_pred", float(test_pred.mean()))


if __name__ == "__main__":
    main()
