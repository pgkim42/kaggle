# Playground Series S6E9 — Predicting Electric Vehicle Purchases

Participation log for [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9). Tabular binary classification: predict `Will_Buy_EV` probability. Metric is **ROC AUC**. Deadline 2026-09-30 23:59 UTC. Public LB is 20% of test.

This is not a winning solution. One change per run; keep only if 5-fold OOF rises.

## Result

Best scored run so far: `exp_lgbm_slow.py`.

| Split | ROC AUC |
| --- | ---: |
| 5-fold OOF | 0.94214 |
| Public leaderboard | **0.94206** |

Submission `55941286` on 2026-09-01.

## Runs (2026-09-01)

| Script | Change vs previous keep | OOF | Public | Decision |
| --- | --- | ---: | ---: | --- |
| `baseline.py` | sklearn HistGB, raw columns | 0.94154 | 0.94147 | keep (first) |
| `exp_lgbm.py` | same folds/cols, LightGBM | 0.94170 | 0.94154 | keep |
| `exp_lgbm_native.py` | LightGBM native categoricals | 0.94170 | — | discard (flat) |
| `exp_lgbm_interact.py` | `env×subsidy`, `income×subsidy`, `env×income` | 0.94195 | 0.94182 | keep |
| `exp_lgbm_slow.py` | same features, `lr=0.03`, 2000 trees | **0.94214** | **0.94206** | keep |
| `exp_xgb_slow.py` | XGBoost, same features/schedule | 0.94192 | — | discard vs LGBM |
| blend 0.7 LGBM + 0.3 XGB | OOF mix of the two | 0.94222 | 0.94207 | public flat; not the reference |

Native categoricals did nothing (cardinality 2–4). XGBoost lost to the slow LightGBM. The blend’s public tick is noise.

## Task

| | |
| --- | --- |
| Host | Kaggle Playground Series (swag, not a prize competition) |
| Rows | 668,665 train / 286,571 test |
| Target | `Will_Buy_EV` (`Yes` / `No`) |
| Positive rate | ~17.5% |
| Missing values | none in the provided columns |
| Features | 7 numeric, 6 categorical |

Numeric: `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level`.

Categorical: `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`, `Range_Anxiety_Level`.

Univariate AUCs on train: environmental concern 0.844, subsidy 0.718, income 0.670. Age/gender/car count sit at ~0.50. Most of 0.94 is those three columns; the rest is leftover ranking.

## Approach

First file, `baseline.py`: HistGradientBoosting because this machine lacked `libgomp`. Later runs use LightGBM after OpenMP was installed.

Kept recipe in `exp_lgbm_slow.py`:

1. Map `Will_Buy_EV` to `{Yes: 1, No: 0}`.
2. Add `subsidy_yes`, `env_x_subsidy`, `income_x_subsidy`, `env_x_income`.
3. Ordinal-encode categoricals.
4. LightGBM, `learning_rate=0.03`, `n_estimators=2000`, early stopping 50, `max_depth=6`, same 5 stratified folds (`random_state=42`).
5. Average test probabilities across folds.

## Setup

From the repo root. Competition data is **not** committed.

```bash
pip install -r requirements.txt
kaggle competitions download -c playground-series-s6e9 -p playground-series-s6e9/data
python -c "import zipfile; zipfile.ZipFile('playground-series-s6e9/data/playground-series-s6e9.zip').extractall('playground-series-s6e9/data')"
python playground-series-s6e9/exp_lgbm_slow.py
```

Scripts look for `data/train.csv` next to themselves.

## License

MIT for the code. Competition data remains under Kaggle’s rules and is not redistributed here.
