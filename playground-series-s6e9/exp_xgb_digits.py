#!/usr/bin/env python3
"""S6E9 exp: XGB with interactions, numeric digits, train+test frequencies."""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" if (ROOT / "data" / "train.csv").exists() else ROOT
TARGET = "Will_Buy_EV"
ID_COL = "id"
BASELINE_OOF = 0.94371

CAT = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]
NUM = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]


def encode_target(s: pd.Series) -> np.ndarray:
    mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0, True: 1, False: 0}
    y = s.map(mapping)
    if y.isna().any():
        leftover = s[y.isna()].unique()[:10]
        raise ValueError(f"unmapped target values: {leftover}")
    return y.astype(int).to_numpy()


def build_frame(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train.copy()
    te = test.copy()
    for col in CAT:
        mapping = {val: i for i, val in enumerate(tr[col].unique())}
        tr[f"LE_{col}"] = tr[col].map(mapping)
        te[f"LE_{col}"] = te[col].map(mapping)
    tr = tr.drop(columns=CAT)
    te = te.drop(columns=CAT)
    if ID_COL in tr.columns:
        tr = tr.drop(columns=[ID_COL])
    if ID_COL in te.columns:
        te = te.drop(columns=[ID_COL])
    y = tr.pop(TARGET)

    tr["Env_Concern_x_Subsidy"] = tr["Environmental_Concern_Level"] * tr["LE_Subsidy_Available"]
    te["Env_Concern_x_Subsidy"] = te["Environmental_Concern_Level"] * te["LE_Subsidy_Available"]
    tr["Env_Concern_x_Income"] = tr["Environmental_Concern_Level"] * tr["Annual_Income_USD"]
    te["Env_Concern_x_Income"] = te["Environmental_Concern_Level"] * te["Annual_Income_USD"]
    tr["Subsidy_x_Income"] = tr["LE_Subsidy_Available"] * tr["Annual_Income_USD"]
    te["Subsidy_x_Income"] = te["LE_Subsidy_Available"] * te["Annual_Income_USD"]
    tr["Env_Concern_x_Commute"] = tr["Environmental_Concern_Level"] * tr["Daily_Commute_km"]
    te["Env_Concern_x_Commute"] = te["Environmental_Concern_Level"] * te["Daily_Commute_km"]
    tr["Home_Charging_x_Subsidy"] = tr["LE_Home_Charging_Possible"] * tr["LE_Subsidy_Available"]
    te["Home_Charging_x_Subsidy"] = te["LE_Home_Charging_Possible"] * te["LE_Subsidy_Available"]
    tr["Range_Anxiety_x_Subsidy"] = tr["LE_Range_Anxiety_Level"] * tr["LE_Subsidy_Available"]
    te["Range_Anxiety_x_Subsidy"] = te["LE_Range_Anxiety_Level"] * te["LE_Subsidy_Available"]
    tr["Income_per_Concern"] = tr["Annual_Income_USD"] / (tr["Environmental_Concern_Level"] + 1)
    te["Income_per_Concern"] = te["Annual_Income_USD"] / (te["Environmental_Concern_Level"] + 1)
    tr["Commute_per_Concern"] = tr["Daily_Commute_km"] / (tr["Environmental_Concern_Level"] + 1)
    te["Commute_per_Concern"] = te["Daily_Commute_km"] / (te["Environmental_Concern_Level"] + 1)

    digit_cols = list(NUM)
    for col in digit_cols:
        for k in range(-4, 4):
            name = f"{col}_digit{k}"
            scale = 10.0**k
            tr[name] = (tr[col].fillna(0) // scale % 10).astype(np.int8)
            te[name] = (te[col].fillna(0) // scale % 10).astype(np.int8)

    for column in list(tr.columns):
        freq_map = pd.concat([tr[column], te[column]], axis=0).value_counts(normalize=True)
        tr[f"{column}_freq"] = tr[column].map(freq_map).astype(np.float32)
        te[f"{column}_freq"] = te[column].map(freq_map).astype(np.float32)

    return tr, te, encode_target(y)


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    X, X_test, y = build_frame(train, test)
    print("train", train.shape, "X", X.shape, "test", X_test.shape)
    print("model XGB digits+allfreq 5fold")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=float)
    test_pred = np.zeros(len(X_test), dtype=float)
    fold_scores = []

    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        clf = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            learning_rate=0.03,
            max_depth=7,
            min_child_weight=5,
            subsample=0.7,
            colsample_bytree=0.5,
            n_estimators=2000,
            early_stopping_rounds=100,
            n_jobs=-1,
            random_state=42 + fold,
        )
        clf.fit(
            X.iloc[tr],
            y[tr],
            eval_set=[(X.iloc[va], y[va])],
            verbose=False,
        )
        p_va = clf.predict_proba(X.iloc[va])[:, 1]
        p_te = clf.predict_proba(X_test)[:, 1]
        oof[va] = p_va
        test_pred += p_te / cv.n_splits
        auc = roc_auc_score(y[va], p_va)
        fold_scores.append(auc)
        best = getattr(clf, "best_iteration", None)
        print(f"fold {fold} auc={auc:.5f} best_iter={best}", flush=True)

    cv_mean = float(np.mean(fold_scores))
    cv_auc = roc_auc_score(y, oof)
    delta = cv_auc - BASELINE_OOF
    decision = "KEEP" if cv_auc > BASELINE_OOF + 1e-5 else "DISCARD"
    print(f"cv mean={cv_mean:.5f} oof={cv_auc:.5f}")
    print(f"baseline lgbm_allfreq oof={BASELINE_OOF:.5f}")
    print(f"delta_oof={delta:+.5f} decision={decision}")

    np.save(DATA / "oof_xgb_digits.npy", oof)
    out = DATA / "submission_xgb_digits.csv"
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred}).to_csv(out, index=False)
    print("wrote", out, "mean_pred", float(test_pred.mean()))


if __name__ == "__main__":
    main()
