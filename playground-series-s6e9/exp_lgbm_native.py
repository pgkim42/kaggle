#!/usr/bin/env python3
"""S6E9 exp: LightGBM native categoricals. Same folds/cols as exp_lgbm."""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" if (ROOT / "data" / "train.csv").exists() else ROOT
TARGET = "Will_Buy_EV"
ID_COL = "id"
BASELINE_OOF = 0.94170

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


def as_fold_frame(src: pd.DataFrame, cat_levels: dict[str, pd.Index]) -> pd.DataFrame:
    out = src.copy()
    for c, levels in cat_levels.items():
        out[c] = pd.Categorical(out[c], categories=levels)
    return out


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = encode_target(train[TARGET])
    X = train[NUMERIC + CATEGORICAL]
    X_test = test[NUMERIC + CATEGORICAL]

    print("train", train.shape, "test", test.shape, "pos_rate", float(y.mean()))
    print("model LightGBM native cats n_estimators=250 lr=0.08 max_depth=6")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(len(test), dtype=float)
    fold_scores = []

    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        X_tr = X.iloc[tr]
        cat_levels = {c: pd.Index(X_tr[c].astype("string").unique()) for c in CATEGORICAL}
        X_tr = as_fold_frame(X_tr, cat_levels)
        X_va = as_fold_frame(X.iloc[va], cat_levels)
        X_te = as_fold_frame(X_test, cat_levels)
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
            eval_X=X_va,
            eval_y=y[va],
            eval_metric="auc",
            categorical_feature=CATEGORICAL,
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
    delta = cv_auc - BASELINE_OOF
    decision = "KEEP" if cv_auc > BASELINE_OOF else "DISCARD"
    print(f"cv mean={cv_mean:.5f} oof={cv_auc:.5f}")
    print(f"baseline lgbm-ordinal oof={BASELINE_OOF:.5f}")
    print(f"delta_oof={delta:+.5f} decision={decision}")

    out = DATA / "submission_lgbm_native.csv"
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred}).to_csv(out, index=False)
    print("wrote", out, "rows", len(test_pred), "mean_pred", float(test_pred.mean()))


if __name__ == "__main__":
    main()
