# kaggle

Competition write-ups and scored baselines. One folder per contest. Data stays on Kaggle; this repo only keeps code and notes.

I am using Playground Series as the on-ramp — monthly tabular problems, no prize money, same submission loop as the rest of the site.

## Competitions

| Folder | Contest | Task | Metric | Public LB | Notes |
| --- | --- | --- | ---: | ---: | --- |
| [`playground-series-s6e9`](playground-series-s6e9/) | [S6E9 — Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9) | binary classification | ROC AUC | 0.94147 | first submission, sklearn HistGB, 2026-09-01 |

## Layout

```
playground-series-s6e9/   # slug from the Kaggle URL
  README.md               # problem, score, what I actually tried
  baseline.py             # the run that was submitted
```

Later contests follow the same slug. Shared Python deps live in `requirements.txt` at the root until a contest needs its own.

## Setup

Token from [Kaggle API settings](https://www.kaggle.com/settings/api):

```bash
mkdir -p ~/.kaggle
echo 'KGAT_...' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Each contest README has the download / train / submit commands for that dataset.

## License

MIT for the code. Competition data is not redistributed.
