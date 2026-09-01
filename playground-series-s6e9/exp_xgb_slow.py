#!/usr/bin/env python3
"""S6E9 exp: XGBoost on the same interact features / folds as lgbm_slow."""

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" if (ROOT / "data" / "train.csv").exists() else ROOT
TARGET = "Will_Buy_EV"
ID_COL = "id"
BASELINE_OOF = 0.94214

NUMERIC = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "subsidy_yes",
    "env_x_subsidy",
    "income_x_subsidy",
    "env_x_income",
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


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    subsidy = (out["Subsidy_Available"] == "Yes").astype(np.float64)
    env = out["Environmental_Concern_Level"].astype(np.float64)
    income = out["Annual_Income_USD"].astype(np.float64)
    out["subsidy_yes"] = subsidy
    out["env_x_subsidy"] = env * subsidy
    out["income_x_subsidy"] = income * subsidy
    out["env_x_income"] = env * (income / 1e5)
    return out


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = encode_target(train[TARGET])
    X = add_interactions(train)[NUMERIC + CATEGORICAL]
    X_test = add_interactions(test)[NUMERIC + CATEGORICAL]

    print("train", train.shape, "test", test.shape)
    print("model XGB interact lr=0.03 n_estimators=2000 early_stopping=50")

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

    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        X_tr = prep.fit_transform(X.iloc[tr])
        X_va = prep.transform(X.iloc[va])
        X_te = prep.transform(X_test)
        clf = xgb.XGBClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=40,
            subsample=1.0,
            colsample_bytree=1.0,
            tree_method="hist",
            eval_metric="auc",
            early_stopping_rounds=50,
            n_jobs=-1,
            random_state=42,
        )
        clf.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], verbose=False)
        best = int(clf.best_iteration)
        p_va = clf.predict_proba(X_va)[:, 1]
        p_te = clf.predict_proba(X_te)[:, 1]
        oof[va] = p_va
        test_pred += p_te / cv.n_splits
        auc = roc_auc_score(y[va], p_va)
        fold_scores.append(auc)
        print(f"fold {fold} auc={auc:.5f} best_iter={best}", flush=True)

    cv_mean = float(np.mean(fold_scores))
    cv_auc = roc_auc_score(y, oof)
    delta = cv_auc - BASELINE_OOF
    decision = "KEEP" if cv_auc > BASELINE_OOF + 1e-5 else "DISCARD"
    print(f"cv mean={cv_mean:.5f} oof={cv_auc:.5f}")
    print(f"baseline lgbm_slow oof={BASELINE_OOF:.5f}")
    print(f"delta_oof={delta:+.5f} vs_lgbm={decision}")

    np.save(DATA / "oof_xgb_slow.npy", oof)
    out = DATA / "submission_xgb_slow.csv"
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred}).to_csv(out, index=False)
    print("wrote", out, "mean_pred", float(test_pred.mean()))

    lgbm_oof_path = DATA / "oof_lgbm_slow.npy"
    lgbm_sub_path = DATA / "submission_lgbm_slow.csv"
    if lgbm_oof_path.exists() and lgbm_sub_path.exists():
        lgbm_oof = np.load(lgbm_oof_path)
        lgbm_te = pd.read_csv(lgbm_sub_path)[TARGET].to_numpy()
        print("lgbm_oof", roc_auc_score(y, lgbm_oof))
        for w in (0.3, 0.4, 0.5, 0.6, 0.7):
            blend = w * lgbm_oof + (1.0 - w) * oof
            print(f"blend lgbm={w:.1f} xgb={1-w:.1f} oof={roc_auc_score(y, blend):.5f}")
        w = 0.6
        blend_oof = roc_auc_score(y, w * lgbm_oof + (1.0 - w) * oof)
        blend_te = w * lgbm_te + (1.0 - w) * test_pred
        blend_delta = blend_oof - BASELINE_OOF
        blend_decision = "KEEP" if blend_oof > BASELINE_OOF + 1e-5 else "DISCARD"
        print(f"chosen blend 0.6/0.4 oof={blend_oof:.5f} delta={blend_delta:+.5f} {blend_decision}")
        pd.DataFrame({ID_COL: test[ID_COL], TARGET: blend_te}).to_csv(
            DATA / "submission_blend_lgbm_xgb.csv", index=False
        )


if __name__ == "__main__":
    main()
