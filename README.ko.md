# kaggle

캐글 대회에 나간 기록이다. 대회마다 폴더 하나. 여기 있는 건 코드와 메모뿐이고, **데이터 파일은 올리지 않는다.** (캐글 규칙)

캐글을 처음 해 보는 사람이 매달 열리는 Playground Series로 연습하는 저장소다. 상금은 없고, 제출 방식은 큰 대회와 같다.

[English](README.md)

## 이 저장소가 뭐냐

캐글은 표(엑셀 같은 것)를 주고 “이 사람이 전기차를 살까?”처럼 맞혀 보라고 하는 사이트다. 답을 CSV로 올리면 점수가 나온다.

이 리포는 “1등 솔루션”이 아니다. **무엇을 바꿨고, 점수가 올랐는지 떨어졌는지**를 남긴다. 머신러닝을 몰라도 “참가 기록”으로 읽히게 쓰는 게 목적이다.

## 참가한 대회

| 폴더 | 대회 | 하는 일 | 점수 방식 | 공개 점수 | 메모 |
| --- | --- | --- | ---: | ---: | --- |
| [`playground-series-s6e9`](playground-series-s6e9/) | [S6E9 — 전기차 구매 예측](https://www.kaggle.com/competitions/playground-series-s6e9) | 살지 말지 | ROC AUC | 0.94206 | 2026-09-01, HistGB에서 LightGBM까지 |

대회 설명이 필요하면 [한국어 대회 노트](playground-series-s6e9/README.ko.md)부터 읽으면 된다.

## 폴더 구조

```
playground-series-s6e9/   # 캐글 주소의 이름 그대로
  README.md               # 영어 기록
  README.ko.md            # 한국어, 초보용
  baseline.py             # 첫 제출
  exp_*.py                # 그다음 실험
```

다음 대회는 `playground-series-s6e10/`처럼 폴더만 추가한다. 공통 패키지는 루트 `requirements.txt`.

## 준비

토큰은 [캐글 API 설정](https://www.kaggle.com/settings/api)에서 받는다.

```bash
mkdir -p ~/.kaggle
echo 'KGAT_...' > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

데이터 받기·학습·제출 명령은 각 대회 README에 있다.

## 라이선스

코드는 MIT. 대회 데이터는 재배포하지 않는다.
