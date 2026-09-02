#!/usr/bin/env python3
"""S6E9 exp: freq recipe + fold-safe smoothed target encoding on income/commute."""

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
BASELINE_OOF = 0.94332
TE_SMOOTH = 20.0

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
    "income_floor",
    "commute_floor",
    "income_freq",
    "commute_freq",
    "income_te",
    "commute_te",
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


def add_static(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    subsidy = (out["Subsidy_Available"] == "Yes").astype(np.float64)
    env = out["Environmental_Concern_Level"].astype(np.float64)
    income = out["Annual_Income_USD"].astype(np.float64)
    commute = out["Daily_Commute_km"].astype(np.float64)
    out["subsidy_yes"] = subsidy
    out["env_x_subsidy"] = env * subsidy
    out["income_x_subsidy"] = income * subsidy
    out["env_x_income"] = env * (income / 1e5)
    out["income_floor"] = (income == 30000.0).astype(np.float64)
    out["commute_floor"] = (commute == 5.0).astype(np.float64)
    return out


def add_fold_stats(base: pd.DataFrame, src: pd.DataFrame, y_src: np.ndarray) -> pd.DataFrame:
    out = base.copy()
    inc_map = src["Annual_Income_USD"].value_counts()
    com_map = src["Daily_Commute_km"].value_counts()
    out["income_freq"] = out["Annual_Income_USD"].map(inc_map).fillna(1).astype(np.float64)
    out["commute_freq"] = out["Daily_Commute_km"].map(com_map).fillna(1).astype(np.float64)

    gmean = float(y_src.mean())
    stats = src.assign(_t=y_src)
    for col, dest in (("Annual_Income_USD", "income_te"), ("Daily_Commute_km", "commute_te")):
        g = stats.groupby(col)["_t"].agg(["sum", "count"])
        te = (g["sum"] + TE_SMOOTH * gmean) / (g["count"] + TE_SMOOTH)
        out[dest] = out[col].map(te).fillna(gmean).astype(np.float64)
    return out


def main() -> None:
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    y = encode_target(train[TARGET])
    train_s = add_static(train)
    test_s = add_static(test)
    cols = NUMERIC + CATEGORICAL

    print("train", train.shape, "test", test.shape)
    print("model freq LGBM + fold-safe income/commute target encoding m=20")

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

    for fold, (tr, va) in enumerate(cv.split(train_s, y), start=1):
        src = train_s.iloc[tr]
        y_src = y[tr]
        X_tr = add_fold_stats(src, src, y_src)[cols]
        X_va = add_fold_stats(train_s.iloc[va], src, y_src)[cols]
        X_te = add_fold_stats(test_s, src, y_src)[cols]
        Xt = prep.fit_transform(X_tr)
        Xv = prep.transform(X_va)
        Xe = prep.transform(X_te)
        clf = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.03,
            max_depth=6,
            min_child_samples=40,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(
            Xt,
            y[tr],
            eval_X=Xv,
            eval_y=y[va],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        p_va = clf.predict_proba(Xv)[:, 1]
        p_te = clf.predict_proba(Xe)[:, 1]
        oof[va] = p_va
        test_pred += p_te / cv.n_splits
        auc = roc_auc_score(y[va], p_va)
        fold_scores.append(auc)
        print(f"fold {fold} auc={auc:.5f} best_iter={clf.best_iteration_}", flush=True)

    cv_mean = float(np.mean(fold_scores))
    cv_auc = roc_auc_score(y, oof)
    delta = cv_auc - BASELINE_OOF
    decision = "KEEP" if cv_auc > BASELINE_OOF + 1e-5 else "DISCARD"
    print(f"cv mean={cv_mean:.5f} oof={cv_auc:.5f}")
    print(f"baseline lgbm_freq oof={BASELINE_OOF:.5f}")
    print(f"delta_oof={delta:+.5f} decision={decision}")

    np.save(DATA / "oof_lgbm_te.npy", oof)
    out = DATA / "submission_lgbm_te.csv"
    pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_pred}).to_csv(out, index=False)
    print("wrote", out, "mean_pred", float(test_pred.mean()))


if __name__ == "__main__":
    main()
