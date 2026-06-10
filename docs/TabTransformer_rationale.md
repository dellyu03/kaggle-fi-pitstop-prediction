# TabTransformer 실험 설계 근거
## 데이터 시각화 분석 → 실험 전략 도출 과정

---

## 1. 문제 정의 — 타깃 불균형 확인

데이터 시각화의 첫 번째 분석인 타깃 분포 확인에서, `PitNextLap=0(피트 없음)`이 약 80%, `PitNextLap=1(피트)`이 약 20%로 **4:1 클래스 불균형**이 존재함을 확인했다.

```
단순 정확도 기준으로 모든 샘플을 0으로 예측해도 80% 정확도가 나오는 구조.
→ 정확도가 아닌 AUC / F1을 평가지표로 채택해야 하며,
  학습 시 양성 클래스에 가중치를 부여해야 한다.
```

**설계 반영:**
- 손실 함수에 `pos_weight = (1 - 0.199) / 0.199 ≈ 4.03` 적용
- fold별 train 분포에 따라 동적으로 계산 (`y_tr.mean()` 기반)
- 평가지표: OOF AUC 기준

---

## 2. 피처 분류 — "무엇을 Transformer에 넣을 것인가"

TabTransformer의 핵심 아이디어는 **범주형 피처는 Attention으로, 연속형 피처는 MLP로** 처리하는 것이다. 따라서 피처를 어떻게 분류하느냐가 실험 설계의 출발점이다.

데이터 시각화에서 피처를 이진형 / 명목형 / 순서형 / 연속형으로 분류한 결과를 바탕으로 다음과 같이 판단했다.

---

## 3. 명목형 피처 — Compound, Race → Transformer 입력

### 시각화 근거
명목형 피처별 `PitNextLap` 비율 막대그래프에서:

- **Compound별**: SOFT > MEDIUM > HARD > WET > INTERMEDIATE 순으로 피트 비율이 상이하게 나타남.  
  컴파운드 종류 자체가 타이어 수명 전략을 결정하므로, 단순 정수 인코딩(LabelEncoding)으로는 이 순서 관계를 표현할 수 없다.

- **Race별**: 서킷에 따라 피트 비율이 뚜렷하게 갈린다. 모나코처럼 피트가 불리한 서킷과 타이어 소모가 빠른 서킷은 다른 전략 패턴을 보인다.

```
명목형 피처는 고유값 간에 순서가 없으므로 LabelEncoding의 숫자 크기가 의미 없다.
Embedding을 통해 의미 공간에서 유사한 전략끼리 가깝게 배치하는 표현 학습이 필요하다.
```

**설계 반영:**
- Compound (5종) → Embedding dim 8, Transformer 입력
- Race (26종) → Embedding dim 16, Transformer 입력

---

## 4. 순서형 피처 — Stint, Year → Transformer 입력

### 시각화 근거
순서형 피처별 `PitNextLap` 비율 막대그래프에서:

- **Stint별**: Stint가 높아질수록 피트 확률이 증가하는 **비선형 패턴**이 관찰된다.  
  Stint=1에서는 피트 비율이 낮고, Stint=2~3 이후 급격히 높아지며, 고 Stint에서는 다시 낮아진다(레이스 막판 피트 필요 없음). 이 비선형 관계는 단순 연속형 처리로 포착하기 어렵다.

- **Year별**: 연도별 피트 비율이 다르다. F1 규정 변경(타이어 규격, 의무 피트 스탑 수)이 시즌마다 달라지므로, 특정 연도는 구조적으로 피트 빈도가 다르다.

```
순서형이지만 값 사이의 간격이 동일하지 않고, 피트 예측과의 관계가 비선형이다.
연속형으로 처리하면 선형 변환에 의존하게 되므로, Embedding으로 비선형 패턴을 학습시킨다.
```

**설계 반영:**
- Stint (8종) → Embedding dim 8, Transformer 입력
- Year (4종) → Embedding dim 4, Transformer 입력

---

## 5. 연속형 피처 — MLP 직접 처리

### 시각화 근거
연속형 피처를 5구간으로 구분한 막대그래프에서:

- **TyreLife**: 구간이 높아질수록 피트 비율이 단조 증가하는 **명확한 선형 경향** 존재
- **Cumulative_Degradation**: 누적 열화량이 클수록 피트 확률이 상승
- **RaceProgress**: 레이스 후반부(0.7~1.0) 구간에서 피트 비율이 두드러짐
- **LapTime_Delta**: 음수(랩타임 단축) 구간보다 양수(랩타임 증가, 타이어 노화) 구간에서 피트 비율 상승

```
이들 피처는 값의 크기 자체가 예측에 직접 관여하며, 단조로운 경향을 보인다.
Embedding보다 StandardScaler 정규화 후 MLP에 직접 입력하는 것이 효율적이다.
또한 정확한 수치(예: TyreLife=37.0)가 중요하므로 이산화(Embedding)로 정보를 압축하지 않는다.
```

**설계 반영:**
- LapNumber, TyreLife, Position, LapTime(s), LapTime_Delta, Cumulative_Degradation, RaceProgress, Position_Change, PitStop → StandardScaler 정규화 후 MLP 입력

---

## 6. 이진형 피처 — PitStop → 연속형으로 처리

### 시각화 근거
이진 피처 분포에서 `PitStop`의 피트 비율 차이를 확인했다.  
- `PitStop=1`(이번 랩 피트 발생): 직후 랩에 다시 피트할 가능성이 낮음
- `PitStop=0`: 일반적인 주행 상태

```
2개 값(0/1)이므로 Embedding이 의미 없고(dim=1과 동일),
연속형 입력으로 처리해도 정보 손실이 없다.
```

**설계 반영:**
- PitStop → cont_features에 포함, MLP로 처리

---

## 7. Driver — 의도적 제외

### 시각화 근거 및 판단 근거

데이터 요약에서 Driver는 **887종**으로 전체 피처 중 압도적으로 고카디널리티다.  
명목형 피처 분포 그래프에서 Driver별 피트 비율도 분산이 크게 나타난다.

그러나 다음 이유로 Transformer 입력에서 제외했다:

1. **익명화 문제**: Driver 코드(D109, D086 등)는 익명화되어 시즌/연도에 걸친 동일 드라이버 추적이 불확실하다. 드라이버 전략 패턴이 일관되게 인코딩될 보장이 없다.

2. **계산 비용**: Embedding dim을 충분히 크게 잡지 않으면 887종을 표현하기 어렵고, d_model 확대는 Transformer 전체 파라미터 폭발로 이어진다.

3. **GBDT에서 충분히 처리 가능**: LabelEncoding된 Driver 코드는 GBDT가 분기(split)를 통해 직접 학습할 수 있어, Embedding 없이도 패턴 포착 가능.

```
Driver 정보는 TabTransformer를 거치지 않고 GBDT 단계에서 직접 raw 피처로 활용한다.
```

---

## 8. 검증 전략 — GroupKFold와 데이터 누수 방지

### 시각화 근거

데이터 요약에서 그룹 구조를 확인했다.

- `Race` × `Year` 조합으로 **104개의 독립적인 레이스 단위**가 존재
- 한 레이스 안의 랩들은 **동일한 전략 문맥**을 공유한다 (같은 서킷, 같은 날씨, 같은 팀 전략)

```
예: 2024 Monaco Grand Prix 의 랩 1~78 은 모두 같은 레이스 조건을 가진다.
이 중 랩 40~50 이 train 에, 랩 51~60 이 validation 에 들어가면,
모델은 같은 레이스의 앞 랩을 보고 뒷 랩을 예측하는 셈이 된다.
→ 실제 배포 환경(모르는 레이스)과 전혀 다른 난이도의 검증이 된다.
```

### 왜 일반 KFold는 안 되는가

일반 KFold 또는 랜덤 분할은 **같은 레이스의 랩이 train/validation 양쪽에 섞인다**.

| 분할 방식 | 상황 | 문제 |
|-----------|------|------|
| 랜덤 KFold | 2024 Monaco 의 랩이 train/val 모두 등장 | 같은 레이스 맥락이 train 에 노출 → 검증 AUC 과대 추정 |
| **GroupKFold** | 2024 Monaco 전체가 val 에만 등장 | 모르는 레이스 예측 → 실제 성능에 가까운 검증 |

### GroupKFold 적용 방식

```python
groups = (train['Race'].astype(str) + '_' + train['Year'].astype(str)).values
# 예: 'Monaco Grand Prix_2024', 'British Grand Prix_2023', ...
# → 104개 고유 그룹

gkf = GroupKFold(n_splits=5)
for tr, va in gkf.split(train_df, y, groups=groups):
    # tr: 약 83개 레이스의 모든 랩
    # va: 약 21개 레이스의 모든 랩 (train 과 겹치는 레이스 없음)
```

fold 경계가 **레이스 단위**로 끊기므로, validation은 모델이 한 번도 보지 못한 레이스를 예측하는 시나리오가 된다.

### 데이터 누수 방지 처리

GroupKFold로 인덱스를 나눈 뒤에도 두 가지 누수 위험이 추가로 존재한다.

#### 누수 1: StandardScaler fit 범위

연속형 피처는 정규화가 필요하다. 만약 전체 train 데이터로 Scaler를 fit하면 validation 데이터의 통계(평균·분산)가 Scaler에 반영된다.

```python
# 잘못된 방식 (누수 발생)
sc = StandardScaler().fit(train_df[cont_features])   # val 데이터 통계가 포함됨
tr_df[cont_features] = sc.transform(tr_df[cont_features])
va_df[cont_features] = sc.transform(va_df[cont_features])

# 올바른 방식 (fold별 fit, 현재 적용)
sc = StandardScaler().fit(train_df.iloc[tr][cont_features])  # train fold 만으로 fit
tr_df[cont_features] = sc.transform(tr_df[cont_features])
va_df[cont_features] = sc.transform(va_df[cont_features])
te_df[cont_features] = sc.transform(te_df[cont_features])    # test 도 동일 scaler 사용
```

**fold마다 독립적으로 fit** 함으로써 validation과 test가 train 통계에 영향을 주지 않는다.

#### 누수 2: LabelEncoder fit 범위

범주형 인코딩은 train + test 전체를 합쳐 fit한다.

```python
all_data = pd.concat([train, test], ignore_index=True)
le.fit(all_data[col].astype(str))
```

이는 의도적 처리다. test에만 등장하는 범주(새 드라이버, 새 레이스)가 있을 경우 LabelEncoder가 `unknown` 오류를 낼 수 있으므로, 사전에 전체 범주를 등록해 인코딩 안정성을 확보한다. 타깃값(`PitNextLap`)은 test에 없으므로 이 처리로 인한 정보 누수는 발생하지 않는다.

#### 누수 3: PitStop 피처의 시간적 방향

`PitStop`은 "이번 랩에 피트스탑이 있었는가"를 나타내고, 타깃 `PitNextLap`은 "다음 랩에 피트스탑을 할 것인가"이다.

```
랩 t 의 PitStop (이미 일어난 사건) → 랩 t+1 의 PitNextLap (미래 사건) 예측
```

`PitStop`은 타깃보다 시간적으로 앞서므로 누수가 아니다. 오히려 "방금 피트를 했으면 당분간 하지 않는다"는 유효한 피처다.

---

## 9. 3-모델 앙상블 전략

### 설계 근거

데이터 시각화에서 확인한 피처 특성이 세 가지로 나뉜다:

| 특성 | 주요 피처 | 최적 모델 |
|------|-----------|-----------|
| 범주형 간 상호작용 (Compound × Race × Stint) | 명목형 + 순서형 | TabTransformer |
| 수치의 크기와 임계값 (TyreLife > 30이면 피트) | 연속형 | GBDT |
| 복잡한 비선형 분기 조합 | 전체 피처 | GBDT |

단일 모델로는 양쪽 특성을 동시에 최적으로 처리하기 어렵다. 따라서:

1. **TabTransformer**: Compound × Race × Stint × Year 간 Attention으로 전략 문맥(context) 학습
2. **TabTransformer 임베딩 → LightGBM**: 학습된 문맥 표현을 GBDT의 입력으로 재활용
3. **TabTransformer 임베딩 → XGBoost**: LightGBM과 다른 boosting 방식으로 다양성 확보

```
TabTransformer가 만들어낸 저차원 의미 표현(Embedding)을 
GBDT의 입력으로 연결함으로써, 두 모델의 강점을 계층적으로 결합한다.
```

---

## 9. 전체 설계 흐름 요약

```
[데이터 시각화 분석]
        │
        ├─ 타깃 불균형(80:20) ──────────────→ pos_weight 적용
        │
        ├─ 명목형(Compound, Race)
        │   비선형 범주별 피트 패턴 ─────────→ Embedding + Transformer
        │
        ├─ 순서형(Stint, Year)
        │   비선형·비등간격 패턴 ────────────→ Embedding + Transformer
        │
        ├─ 연속형(TyreLife, Degradation 등)
        │   단조 선형 경향 ──────────────────→ Scaler + MLP 직접 입력
        │
        ├─ 이진형(PitStop)
        │   2값 → Embedding 불필요 ──────────→ cont_features에 포함
        │
        └─ 명목형(Driver, 887종)
            고카디널리티 + 익명화 ───────────→ GBDT raw 입력으로 처리
                                                      │
                                           ┌──────────┴──────────┐
                                     LightGBM              XGBoost
                                           │                     │
                                           └──────────┬──────────┘
                                                  Soft Voting
                                               (가중치 최적화)
```
