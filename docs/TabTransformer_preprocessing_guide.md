# TabTransformer — 전처리 로직 완전 정리

이 문서는 `TabTransformer_CUDA.ipynb`(베이스)와 `TabTransformer_FE.ipynb`(피처 엔지니어링 버전)의
전처리 로직을 단계별로 설명한다.  
설계 결정의 근거는 [`TabTransformer_rationale.md`](TabTransformer_rationale.md) 를 함께 참고.

---

## 전체 흐름 한눈에 보기

```
CSV 로드
  │
  ▼
피처 정의 ──────────────────────────────────────────────────────
  │  cat_features  → Embedding → Transformer → MLP
  │  cont_features → StandardScaler(fold별) → MLP 직접 연결
  │
  ▼
[FE 버전만] 피처 엔지니어링 ─ 수치 상호작용 5개 + 범주형 조합 4개 생성
  │
  ▼
LabelEncoder (train+test 합산 fit)
  │  이유: test에만 등장하는 범주가 있으면 인코딩 오류 발생
  │
  ▼
train_df / test_df 분리
  │
  ▼
타깃(y) · 그룹(groups) · 클래스 가중치(spw) 준비
  │
  ▼
GroupKFold(n_splits=5)  ← groups = Race + '_' + Year
  │
  ├─ Fold마다 반복 ──────────────────────────────────────────────
  │     ① StandardScaler.fit(train fold [cont_features])
  │         → train fold / val fold / test_df 모두 transform
  │     ② TabTransformer 학습
  │     ③ OOF 예측 + 임베딩 추출
  │     ④ LightGBM / XGBoost 학습 (임베딩 입력)
  │
  ▼
Fold bagging → Soft Voting 앙상블 → 제출
```

---

## Step 1. 데이터 로드

```python
train = pd.read_csv('train.csv')   # (439140, 16)
test  = pd.read_csv('test.csv')    # (188165, 15)
```

| 항목 | train | test |
|------|-------|------|
| 행 수 | 439,140 | 188,165 |
| 컬럼 수 | 16 | 15 (`PitNextLap` 없음) |
| 결측값 | 없음 | 없음 |

---

## Step 2. 피처 정의

모든 컬럼(`id`, `PitNextLap` 제외)은 두 경로 중 하나로 분류된다.

### 베이스 버전 (`TabTransformer_CUDA.ipynb`)

| 경로 | 피처 목록 | 개수 |
|------|----------|------|
| **Transformer 입력** (범주형 Embedding) | `Compound`, `Race`, `Stint`, `Year` | 4개 |
| **MLP 직접 입력** (연속형 스케일링) | `LapNumber`, `TyreLife`, `Position`, `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation`, `RaceProgress`, `Position_Change`, `PitStop` | 9개 |

> **`Driver`를 제외한 이유**: 887종으로 고카디널리티. 익명화 코드여서 시즌 간 동일 드라이버
> 연결 불확실. GBDT가 raw LabelEncoding으로 충분히 처리 가능.
> → 자세한 근거: [`TabTransformer_rationale.md § 7`](TabTransformer_rationale.md)

> **`Stint`를 Transformer에 넣은 이유**: Stint가 높아질수록 피트 비율이 비선형으로
> 증감한다(초반↑ → 후반↓). 단순 숫자 크기만으로 표현하면 이 비선형 패턴을 놓친다.

### FE 버전 (`TabTransformer_FE.ipynb`)과의 차이

| 항목 | 베이스 | FE 버전 |
|------|--------|---------|
| cat_features | 4개 | 8개 (`Driver` 추가 + 조합 피처 4개) |
| cont_features | 9개 | 15개 (`Stint` 이동 + 상호작용 피처 5개) |
| `Driver` | 제외 | 포함 (Embedding dim=32) |
| `Stint` | cat (Embedding) | cont (StandardScaler) |

---

## Step 3. 피처 엔지니어링 (FE 버전 전용)

베이스 버전은 이 단계가 없다.

### 3-1. 수치 상호작용 피처 5개

```python
df['TyreLife_x_LapNumber']    = df['TyreLife'] * df['LapNumber']
df['RaceProgress_x_TyreLife'] = df['RaceProgress'] * df['TyreLife']
df['LapDelta_DIV_LapTime']    = df['LapTime_Delta'] / (df['LapTime (s)'] + 1e-6)
df['CumDeg_DIV_TyreLife']     = df['Cumulative_Degradation'] / (df['TyreLife'] + 1e-6)
df['LapNumber_DIV_TyreLife']  = df['LapNumber'] / (df['TyreLife'] + 1e-6)
```

| 피처 | 의미 |
|------|------|
| `TyreLife_x_LapNumber` | 절대적 타이어 소모량 (랩 수 × 타이어 나이) |
| `RaceProgress_x_TyreLife` | 레이스 후반부에 타이어가 오래됐을수록 큰 값 |
| `LapDelta_DIV_LapTime` | 랩타임 변화율의 상대적 크기 (절대값 기준 정규화) |
| `CumDeg_DIV_TyreLife` | 랩당 평균 열화량 (타이어 수명 대비 누적 열화) |
| `LapNumber_DIV_TyreLife` | 타이어 교체 없이 달린 비율 (값이 크면 오래된 타이어) |

> `+ 1e-6` 처리: `TyreLife=0` 또는 `LapTime=0` 인 경우 0 나눗셈 방지.

### 3-2. 범주형 조합 피처 4개

```python
df['Driver_Compound']      = Driver + '_' + Compound
df['Driver_Race']          = Driver + '_' + Race
df['Compound_Race']        = Compound + '_' + Race
df['Driver_Compound_Race'] = Driver + '_' + Compound + '_' + Race
```

| 피처 | 고유값 수 | 의미 |
|------|----------|------|
| `Driver_Compound` | 3,021 | 드라이버별 타이어 전략 성향 |
| `Driver_Race` | 15,408 | 드라이버별 서킷 전략 |
| `Compound_Race` | 100 | 서킷별 타이어 컴파운드 선택 패턴 |
| `Driver_Compound_Race` | 35,737 | 세 가지를 결합한 가장 세부적인 전략 문맥 |

이 조합 피처들은 "이 드라이버가 이 서킷에서 이 타이어를 사용할 때 피트 패턴"을 직접 모델링한다.  
단일 피처만으로는 포착하기 어려운 **교호작용(interaction)**을 LabelEncoding으로 빠르게 생성한다.

---

## Step 4. 범주형 인코딩 — LabelEncoder

```python
# train + test 합쳐 fit → 분리
all_data = pd.concat([train, test], ignore_index=True)
all_data = add_features(all_data)   # FE 버전만 해당

for col in cat_features:
    le = LabelEncoder()
    all_data[col + '_enc'] = le.fit_transform(all_data[col].astype(str))
    # → Compound_enc, Race_enc, Stint_enc, Year_enc, ...

train_df = all_data.iloc[:len(train)].copy()
test_df  = all_data.iloc[len(train):].copy().reset_index(drop=True)
```

**왜 train+test를 합쳐서 fit하나?**

```
만약 train만으로 fit할 경우:
  test에 새 드라이버 'D999' 등장 → LabelEncoder가 처리 불가 → 런타임 오류

해결: 사전에 train+test 전체 범주를 등록해 두면 어떤 값이 와도 안정적으로 인코딩.
타깃값(PitNextLap)은 test에 없으므로 이 처리는 정보 누수가 아니다.
```

**결과**: 인코딩된 컬럼명에 `_enc` 접미사 추가 (원본 컬럼은 유지)

---

## Step 5. 타깃 · 그룹 · 클래스 가중치

```python
y        = train['PitNextLap'].values.astype(np.float32)
groups   = (train['Race'].astype(str) + '_' + train['Year'].astype(str)).values
pos_rate = y.mean()           # ≈ 0.199 (약 20%)
spw      = (1 - pos_rate) / pos_rate   # ≈ 4.03
```

| 변수 | 값 | 용도 |
|------|----|------|
| `y` | float32 배열 (0.0 / 1.0) | 학습 타깃 |
| `groups` | `'Monaco Grand Prix_2024'` 형태 문자열 | GroupKFold 분할 기준 |
| `spw` | ≈ 4.03 | `BCEWithLogitsLoss(pos_weight=...)` 및 GBDT `scale_pos_weight` |

**클래스 가중치 계산 원리**:

```
피트(1) 샘플 1개의 손실 = 비피트(0) 샘플 손실 × 4.03
→ 적은 양성 샘플이 학습에 4배 더 강하게 반영됨
→ 실제 fold별 y_tr.mean()으로 동적 계산 (fold마다 미묘하게 다를 수 있음)
```

---

## Step 6. GroupKFold — 검증 전략

```python
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_df, y, groups=groups))
# splits: 5개 (train_indices, val_indices) 쌍
```

**일반 KFold와 GroupKFold의 차이**:

```
일반 KFold:
  Fold 1 train: [Monaco_랩1, Monaco_랩20, Monaco_랩40, ...]
  Fold 1 val:   [Monaco_랩10, Monaco_랩30, Monaco_랩50, ...]
  → 같은 레이스 랩이 train/val 양쪽에 존재 → 검증 성능 과대 추정

GroupKFold:
  Fold 1 train: [Monaco 2024 전체, Silverstone 2023 전체, ...]
  Fold 1 val:   [Monza 2024 전체, Bahrain 2022 전체, ...]
  → 같은 레이스는 반드시 한쪽에만 → 미지의 레이스 예측 시뮬레이션
```

104개 레이스 × 5-fold → val 1개 fold ≈ 20~21개 레이스

---

## Step 7. Per-fold 연속형 스케일링 (Fold 루프 내부)

이 단계는 **Fold 루프 안**에서 실행된다.

```python
for f, (tr, va) in enumerate(splits):

    # train fold 인덱스만으로 Scaler fit
    sc = StandardScaler().fit(train_df.iloc[tr][cont_features])

    # 세 데이터셋 모두 transform (같은 scaler 사용)
    tr_df[cont_features] = sc.transform(tr_df[cont_features])
    va_df[cont_features] = sc.transform(va_df[cont_features])
    te_df[cont_features] = sc.transform(te_df[cont_features])
```

**왜 전체 train으로 미리 fit하지 않나?**

```
만약 전체 train으로 fit 하면:
  scaler의 평균/분산에 validation 데이터의 통계가 포함됨
  → 모델이 val 데이터를 "미리 본" 것과 동일한 효과 → 검증 AUC 낙관적 추정

올바른 방식: fold별 train 인덱스만으로 fit
  → val도 test도 train의 통계로만 변환
  → 실제 배포 환경 재현 (모르는 레이스의 데이터를 정규화)
```

---

## Step 8. 전처리 후 데이터 형태

각 fold 학습 직전의 데이터 형태:

| 데이터 | 범주형 입력 | 연속형 입력 |
|--------|-----------|-----------|
| 형식 | `LongTensor` (정수 인덱스) | `FloatTensor` (스케일된 실수) |
| 베이스 차원 | `(N, 4)` | `(N, 9)` |
| FE 차원 | `(N, 8)` | `(N, 15)` |

TabTransformer 입력 → 출력 차원:

```
베이스:
  범주형 Embedding → Transformer → flatten: max_emb_dim × 4 = 16 × 4 = 64
  연속형 직접: 9
  MLP 입력 total_dim = 64 + 9 = 73

FE:
  범주형 Embedding → flatten: max_emb_dim × 8 = 32 × 8 = 256
  연속형 직접: 15
  MLP 입력 total_dim = 256 + 15 = 271
```

---

## 두 버전 전처리 비교표

| 항목 | 베이스 (`CUDA.ipynb`) | FE (`FE.ipynb`) |
|------|---------------------|----------------|
| `Driver` | 제외 | 범주형 Embedding (dim=32) |
| `Stint` | 범주형 Embedding (dim=8) | 연속형 StandardScaler |
| 수치 상호작용 피처 | 없음 | 5개 추가 |
| 범주형 조합 피처 | 없음 | 4개 추가 |
| cat_features 수 | 4개 | 8개 |
| cont_features 수 | 9개 | 15개 |
| MLP 입력 차원 | 73 | 271 |
| OOF AUC (TT) | 0.9078 | 약 0.90~0.92 |
| OOF AUC (GBDT 앙상블) | 0.9296 | 0.9271 |

> FE 버전이 TT 단독으로는 더 높을 수 있으나,
> Driver 조합 피처의 고카디널리티가 GBDT 앙상블에 노이즈를 줄 수 있다.

---

## 핵심 설계 원칙 요약

| 원칙 | 적용 |
|------|------|
| **범주형은 Embedding** | 숫자 크기가 의미 없는 피처 (Race, Compound 등) |
| **비선형 순서형도 Embedding** | Stint처럼 피트 확률과 비선형 관계인 피처 |
| **연속형은 fold별 Scaler** | 검증 누수 방지 |
| **LabelEncoder는 train+test 합산 fit** | test 미등장 범주 오류 방지 |
| **GroupKFold (Race+Year)** | 레이스 단위 데이터 누수 방지 |
| **pos_weight 동적 계산** | fold별 클래스 비율 변화 대응 |
