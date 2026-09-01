# Playground Series S6E9 — Predicting Electric Vehicle Purchases

First submitted baseline for [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9).

This is a tabular binary classification problem: given household and commuting features, predict the probability that a person will buy an electric vehicle (`Will_Buy_EV`). The official metric is **ROC AUC**.

This repository is a participation log, not a winning solution. It records the first end-to-end submission: data through Kaggle’s API, a sklearn `HistGradientBoostingClassifier` pipeline, 5-fold stratified CV, and a scored leaderboard file.

## Result

| Split | ROC AUC |
| --- | ---: |
| 5-fold CV (mean) | 0.94156 |
| 5-fold OOF | 0.94154 |
| Public leaderboard | **0.94147** |

- Submission: `55940574` on 2026-09-01
- Competition deadline: 2026-09-30 23:59 UTC
- Public leaderboard uses 20% of the test labels; the private score is hidden until the deadline

The public score matched local OOF to four decimals, which is what you want from a first baseline: the model generalized instead of memorizing the training table.

## Task

| | |
| --- | --- |
| Host | Kaggle Playground Series (practice / swag, not a prize competition) |
| Rows | 668,665 train / 286,571 test |
| Target | `Will_Buy_EV` (`Yes` / `No`) |
| Positive rate | ~17.5% |
| Missing values | none in the provided columns |
| Features | 7 numeric, 6 categorical |

Numeric: `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level`.

Categorical: `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`, `Range_Anxiety_Level`.

The training labels are strings (`Yes`/`No`). Predictions are class probabilities in `[0, 1]`, not hard 0/1 labels. AUC ranks people by predicted purchase probability; a constant “No” predictor would look accurate on a 17.5% positive problem and still score ~0.5.

## Approach

`baseline.py` is a single script.

1. Map `Will_Buy_EV` to `{Yes: 1, No: 0}`.
2. Pass numeric columns through unchanged.
3. Ordinal-encode categoricals (`unknown` / missing → `-1`).
4. Fit `HistGradientBoostingClassifier` with those six columns marked categorical.
5. Average test probabilities across **5 stratified folds**.

LightGBM / XGBoost are the usual Playground defaults. This machine did not have `libgomp`, so the first submission uses sklearn’s histogram gradient boosting instead of adding a system library. Early stopping is on; `random_state=42`.

This is a baseline on purpose: no extra features, no target encoding, no stacking, no public-notebook blend.

Fold scores:

```
fold 1  0.94036
fold 2  0.94124
fold 3  0.94264
fold 4  0.94205
fold 5  0.94148
```

## Setup

Competition data is **not** in this repo. Download it with the [Kaggle API](https://github.com/Kaggle/kaggle-cli) after accepting the rules.

```bash
# token from https://www.kaggle.com/settings/api
mkdir -p ~/.kaggle
echo 'KGAT_...' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

kaggle competitions download -c playground-series-s6e9 -p data
python -c "import zipfile; zipfile.ZipFile('data/playground-series-s6e9.zip').extractall('data')"
```

Put `train.csv` and `test.csv` in `data/` (the script looks there first).

```bash
python baseline.py
kaggle competitions submit -c playground-series-s6e9 \
  -f submission.csv \
  -m "baseline histgb 5fold oof 0.94154"
```

## What’s next

The gap from 0.941 to the early public top (~0.946) is small in absolute AUC and large in rank. Sensible next steps, in order:

1. Inspect which features actually move AUC (permutation / split gain), instead of adding models first.
2. Try a GBDT that handles categoricals natively (LightGBM or CatBoost) once OpenMP is available.
3. Only then consider encoding tricks or a blend, and only if they beat this OOF on every fold.

Until then, this file is the reference run.

## License

Code in this repository is MIT. The competition data remains under Kaggle’s rules and is not redistributed here.
