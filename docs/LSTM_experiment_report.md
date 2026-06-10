# F1 Pit Stop 예측 — LSTM 실험 보고서

## 1. 실험 목표

F1 레이스 랩 데이터를 기반으로 **"다음 랩에 피트스탑을 할 것인가(PitNextLap)"** 를 예측하는 이진 분류 모델을 LSTM으로 구현한다.  
시계열 특성(랩 순서)을 슬라이딩 윈도우로 포착하고, CUDA 환경에서 Mixed Precision을 활용해 학습 효율을 극대화한다.

---

## 2. 데이터셋 개요

| 항목 | 내용 |
|------|------|
| 학습 데이터 | train.csv |
| 테스트 데이터 | test.csv |
| 타깃 변수 | `PitNextLap` (0: 피트 없음, 1: 다음 랩 피트) |
| 양성 비율 | 약 26.13% (클래스 불균형 존재) |
| 샘플 수 (윈도우 후) | 263,412개 |

### 2-1. 피처 목록

| 피처명 | 타입 | 처리 방식 | 설명 |
|--------|------|-----------|------|
| `LapNumber` | 연속형 | StandardScaler | 현재 랩 번호 |
| `TyreLife` | 연속형 | StandardScaler | 타이어 사용 랩 수 |
| `LapTime (s)` | 연속형 | StandardScaler | 해당 랩 소요 시간(초) |
| `LapTime_Delta` | 연속형 | StandardScaler | 이전 랩 대비 랩타임 변화량 |
| `Cumulative_Degradation` | 연속형 | StandardScaler | 누적 타이어 열화량 |
| `RaceProgress` | 연속형 | StandardScaler | 전체 레이스 진행률 (0~1) |
| `Position` | 연속형 | StandardScaler | 현재 순위 |
| `Position_Change` | 연속형 | StandardScaler | 직전 랩 대비 순위 변화 |
| `Stint` | 연속형 | StandardScaler | 현재 스틴트 번호 (피트 타이밍 직결) |
| `PitStop` | 연속형 | StandardScaler | 해당 랩 피트스탑 여부 (직전 피트 이력) |
| `Driver` | 명목형 | Embedding (dim=16) | 드라이버 ID (887종) |
| `Compound` | 명목형 | Embedding (dim=3) | 타이어 컴파운드 (5종) |
| `Race` | 명목형 | Embedding (dim=6) | 레이스명 (26종) |
| `Year` | 순서형 | Embedding (dim=2) | 시즌 연도 (2022~2025) |

---

### 2-2. 피처 선택 전략 및 근거

#### 원칙: 식별자·타깃 외 전수 포함

데이터셋에는 총 16개 컬럼이 존재한다. 이 중 `id`(단순 식별자)와 `PitNextLap`(타깃)을 제외한 **14개 피처 전체**를 입력으로 사용하였다.  
별도의 피처 제거 없이 전수 포함한 이유는 아래와 같다.

1. **결측값 없음** — 모든 컬럼이 439,140행 × 0% 결측으로 클린하다.
2. **피처 수가 적음** — 14개는 LSTM 입력 차원 과부하를 일으킬 규모가 아니다.
3. **도메인상 무관한 컬럼이 없음** — 각 컬럼이 F1 피트스탑 결정 맥락과 직접 연결된다 (아래 상세 근거 참조).

#### 각 피처의 포함 근거

| 피처 | 포함 근거 |
|------|----------|
| `TyreLife` | 피트 결정의 핵심 요인. 타이어가 오래될수록 교체 확률이 높아지며, 단독으로도 강한 예측력을 가진다. |
| `Cumulative_Degradation` | `TyreLife`가 "몇 랩 썼냐"라면, 이 값은 실제 성능 저하량을 누적한다. TyreLife와 상호보완적. |
| `LapTime (s)` · `LapTime_Delta` | 타이어 열화는 랩타임 증가로 나타난다. 절댓값(`LapTime`)과 변화율(`Delta`) 모두 포함하여 절대 성능과 추세를 동시에 포착. |
| `Stint` · `LapNumber` | `Stint`는 "몇 번째 타이어 세트인가"로 전략적 피트 횟수를 반영하고, `LapNumber`는 레이스 전체에서의 타이밍을 나타낸다. 두 피처가 함께 있어야 "초반 공격적 전략 vs 후반 보존 전략"을 구분 가능. |
| `RaceProgress` | `LapNumber`는 레이스마다 총 랩 수가 달라 절대값이 다르다. `RaceProgress`(0~1 정규화)는 이를 보완하여 레이스 간 비교 가능한 진행률을 제공. |
| `Position` · `Position_Change` | 선두권 드라이버는 트래픽 회피를 위해 조기 피트를 선택하는 경향이 있다. 현재 순위와 순위 변화 추세가 전략 압박의 크기를 나타낸다. |
| `PitStop` | **당 랩의 피트 여부**(이미 일어난 사건)이므로 타깃(`PitNextLap`, 미래 사건)과 시간적으로 다르다. 직전 피트 이력으로서 "방금 피트를 했으면 당분간 하지 않는다"는 정보를 담아 데이터 누수 없이 유효한 피처다. |
| `Driver` | 드라이버마다 팀·전략 스타일이 다르다(예: 레드불의 공격적 언더컷 vs 머세데스의 보수적 전략). Embedding을 통해 고카디널리티(887종)를 저차원 벡터로 압축. |
| `Compound` | 소프트 타이어는 빠르지만 마모가 심해 조기 피트 가능성이 높고, 하드 타이어는 반대다. 컴파운드 종류(5종)는 피트 주기를 직접 결정하는 변수. |
| `Race` | 서킷 특성(고속 서킷 vs 저속 시가지)에 따라 타이어 마모율과 피트 전략이 달라진다. 26개 레이스를 Embedding으로 표현하여 서킷별 패턴 학습. |
| `Year` | 시즌마다 타이어 규정과 차량 성능이 변화한다. 연도 효과를 Embedding(dim=2)으로 처리해 규정 변화에 따른 전략 패턴 차이를 반영. |

#### 연속형 vs. Embedding 분류 기준

| 처리 방식 | 적용 대상 | 이유 |
|-----------|----------|------|
| StandardScaler | 수치형 피처 10개 | LSTM 입력의 스케일 통일 필요. 값 간 거리가 의미 있는 연속량. |
| Embedding | `Driver`, `Compound`, `Race`, `Year` | 카테고리 간 순서·거리 관계가 없거나 고카디널리티여서 One-Hot 시 차원 폭발. Embedding은 학습 가능한 밀집 표현을 생성. |

`Year`는 숫자(2022~2025)이지만 연속형 처리를 하지 않은 이유는, 연도 간 거리(예: 2022→2023)가 전략 변화를 선형적으로 나타내지 않기 때문이다. 각 시즌은 별개의 규정 체계를 가지므로 Embedding이 적합하다.

`Stint`와 `Position`은 순서형(각각 1~8, 1~20)이지만 연속형으로 처리하였다. 두 값 모두 크기의 의미(클수록 스틴트가 많다 / 순위가 낮다)가 있어 LSTM 입력에서 스케일만 맞추면 충분하다고 판단했다.

---

## 3. 데이터 전처리

### 3-1. 정렬

```
Driver → Race → Year → LapNumber 순으로 오름차순 정렬
```

시계열 윈도우 구성 전 올바른 랩 순서를 보장하기 위한 선행 작업.

### 3-2. 범주형 인코딩

- `train + test` 를 합쳐 `LabelEncoder.fit()` → 테스트에 미등장 범주 방지
- 인코딩된 컬럼명: `Driver_enc`, `Compound_enc`, `Race_enc`, `Year_enc`

### 3-3. 슬라이딩 윈도우

`(Driver, Race, Year)` 단위 그룹 내에서 연속 `window_size` 랩을 묶어 하나의 샘플로 구성.

```
window_size = 5 기준:

입력: [lap_t-4, lap_t-3, lap_t-2, lap_t-1, lap_t]  (5 timesteps)
타깃: lap_{t+1} 의 PitNextLap (0 or 1)
```

- 그룹 경계를 넘지 않으므로 레이스 간 정보 혼입 없음
- 결과 shape: `Xn (263412, 5, 10)` / `Xc (263412, 5, 4)`

### 3-4. 연속형 스케일링

- `StandardScaler`를 **fold별 train 데이터에만 fit** → validation 데이터 누수 방지
- 전체 재학습 시에는 전체 train으로 fit

---

## 4. 모델 구조 — PitLSTM

```
[입력층]
  연속형 피처 (10차원)  ──────────────────────────────────────┐
  범주형 피처 (4개)  →  Embedding → concat (27차원)  ─────────┤
                                                              ↓
                                              입력 dim = 10 + 27 = 37
                                                              ↓
[시계열 처리]
  LSTM (37 → 128, num_layers=2, batch_first=True)
  → 마지막 timestep hidden state만 사용 (out[:, -1, :])
                                                              ↓
[출력층]
  Dropout(0.1)
  → Linear(128 → 1)
  → BCEWithLogitsLoss (출력은 logit, sigmoid는 추론 시 적용)
```

| 파라미터 | 값 |
|---------|-----|
| LSTM hidden | 128 |
| LSTM layers | 2 |
| Dropout | 0.1 |
| 총 학습 파라미터 | ~235,000개 |

---

## 5. 학습 전략

### 5-1. 클래스 불균형 대응

```python
BCEWithLogitsLoss(pos_weight=4.03)
```

양성(피트) 비율이 26%이므로 양성 샘플 손실에 4.03배 가중치 부여.

### 5-2. GroupKFold 교차 검증

- `groups = "Race_Year"` 단위로 분리
- 동일 레이스가 train/validation에 동시에 등장하지 않도록 차단
- OOF(Out-of-Fold) 예측으로 신뢰도 높은 검증 수행

### 5-3. EarlyStopping

```python
patience = max(1, int(epochs * 0.3))
```

validation loss 기준, 개선 없으면 조기 종료 → 과적합 방지.

### 5-4. Mixed Precision (AMP)

```python
torch.autocast(device_type='cuda') + GradScaler
```

FP16 연산으로 GPU 메모리 절약 및 처리량 향상.  
실제 처리량: **~62,000 samples/sec** (RTX 5060 Ti 기준)

---

## 6. 하이퍼파라미터 탐색 (HPO)

전체 데이터에서 4만 샘플 추출, 2-fold로 빠르게 비교.

| epochs | batch_size | OOF AUC | 학습 시간 | throughput |
|--------|------------|---------|----------|-----------|
| **20** | **256** | **0.7344** | 13.3s | 28,643/s |
| 20 | 512 | 0.7326 | 8.1s | 46,791/s |
| 40 | 512 | 0.7315 | 14.4s | 47,098/s |
| 40 | 256 | 0.7233 | 15.5s | 39,977/s |

**선택: epochs=20, batch_size=256**

---

## 7. 최종 실험 결과

### 7-1. 최종 5-fold 성능 (전체 데이터)

| 지표 | 값 |
|-----|-----|
| OOF AUC | 0.7587 |
| Fold mean AUC | 0.7599 |
| Fold std AUC | 0.0277 |
| F1 Score | 0.5403 |
| 학습 시간 | 145.8s |
| Peak GPU Memory | 409.1 MB |

### 7-2. 윈도우 크기 민감도 분석

| window | OOF AUC | F1 | 학습 시간 |
|--------|---------|-----|---------|
| 3 | **0.7678** | 0.5209 | 183.6s |
| 5 | 0.7564 | 0.5330 | 121.1s |
| 10 | 0.7636 | **0.6082** | 90.8s |

- AUC 기준 **window=3** 이 최고
- F1 기준 **window=10** 이 최고
- 최종 제출은 window=5 사용

---

## 8. 추론 및 제출

- 전체 train 데이터로 재학습 (best params)
- 테스트 데이터의 초반 랩(window 미충족 시) → 첫 번째 랩 데이터로 **앞 패딩** 처리
- 출력: `submission_lstm_cuda.csv` (188,165행, id + PitNextLap 확률)

---

## 9. 주요 설계 결정 요약

| 결정 | 선택 | 이유 |
|------|------|------|
| 시계열 표현 | 슬라이딩 윈도우 | 구현 단순, 배치 학습 용이 |
| 범주형 처리 | Embedding | Driver 887종 고카디널리티 처리 |
| 검증 전략 | GroupKFold (Race_Year) | 레이스 단위 데이터 누수 방지 |
| 불균형 처리 | pos_weight | 재샘플링 없이 손실함수에서 처리 |
| 스케일링 | fold별 fit | validation 누수 방지 |
| 연산 최적화 | AMP (FP16) | GPU 메모리 절약 + 처리량 향상 |
