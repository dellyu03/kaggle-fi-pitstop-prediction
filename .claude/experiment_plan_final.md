# F1 Pit Stop 예측 — 최종 실험 계획서 (Final)

> Kaggle Playground Series S6E5 · 타겟 `PitNextLap` (이진 분류) · 평가지표 ROC-AUC
> 본 문서는 `experiment_plan.md`(6개 실험 비교 초안)에 수업 노트북
> `C289039_유승환_최적화.ipynb`("효과적으로 학습하기")에서 다룬 **학습 최적화 기법**
> (하이퍼파라미터 탐색 · Dropout · EarlyStopping · train/val 분리)을 정식 단계로 통합한 최종본이다.

---

## 0. 초안 대비 변경 요약 (draft → final)

| 항목 | 초안 (experiment_plan.md) | 최종본 (반영 근거) |
|------|---------------------------|--------------------|
| CV 전략 | StratifiedKFold(5) 전 실험 동일 | **GroupKFold(Race+Year) 5-fold** — 같은 레이스의 랩이 train/val에 섞이는 그룹 누수 차단, 공정 비교 유지 |
| 하이퍼파라미터 | 고정값 명시 | **HPO 단계 정식화** — LSTM은 `GridSearchCV`(수업 기법), GBDT/TabTransformer는 `Optuna`로 확장 |
| 정규화 | 미명시 | **Dropout + EarlyStopping** 신경망 모델 전체 적용 (수업 노트북 기법) |
| 학습 종료 | epochs 고정 | **EarlyStopping(monitor=val AUC, patience)** → best epoch 재학습 패턴 |
| 평가 | OOF AUC 표 | OOF AUC + fold 분산 + 학습시간 + **OOF correlation/diversity** 명시 |

---

## 1. 실험 목적

수업에서 다룬 딥러닝 모델(LSTM)과 tabular 특화 모델(LightGBM, XGBoost, TabTransformer),
그리고 앙상블 전략을 **동일한 CV·피처·시드 조건**에서 비교해, 각 모델 유형의 성능 차이를
ROC-AUC(OOF 기준)로 정량 측정한다. 동시에 수업에서 학습한 **학습 최적화 기법(HPO·정규화·조기종료)**의
효과를 실험적으로 확인한다.

---

## 2. 데이터셋 요약

| 항목 | 내용 |
|------|------|
| 출처 | Kaggle PS S6E5 (원본 F1 Strategy Dataset) |
| 학습 데이터 | 439,140행 × 15 피처 (+타겟) |
| 타겟 | `PitNextLap` (이진) |
| 평가지표 | ROC-AUC |
| 결측치 | 없음 |
| 클래스 불균형 | positive rate ≈ 0.199 → `pos_weight`/`scale_pos_weight` 적용 |
| 범주형 | Driver(887), Compound(5), Race(26) |
| 외부 데이터 | 원본 F1 데이터셋 concat 허용 (전 실험 동일 적용) |

---

## 3. 공통 실험 조건 (전 실험 고정 — 공정 비교 보장)

```
CV 전략   : GroupKFold(n_splits=5), group = Race + Year
            (그룹 누수 차단. fold별 positive rate 모니터링)
평가지표   : ROC-AUC (OOF 기준)
시드       : 42 고정 (numpy / torch / lgbm / xgb / keras)
피처셋     : 전 모델 동일 피처 사용 (아래 4-1)
불균형 대응 : LSTM/TT → pos_weight(BCEWithLogitsLoss),
            GBDT → scale_pos_weight = (neg/pos) ≈ 4.03
디바이스   : MPS(Apple Silicon) 자동 감지, XGBoost는 CPU 강제
```

### 3-1. 공통 피처

```python
# 연속형
num_features = ['Year','LapNumber','Stint','TyreLife','Position',
                'LapTime (s)','LapTime_Delta','Cumulative_Degradation',
                'RaceProgress','Position_Change']
# 범주형
cat_features = ['Driver','Compound','Race']
# 피처 엔지니어링 (GBDT/TT): arithmetic interaction 5개 + cat 조합 4개
```

---

## 4. "효과적 학습" 기법 적용 (← 최적화 노트북 통합)

**LSTM은 수업 노트북(`최적화.ipynb`)의 학습 레시피를 그대로 따른다.** GBDT/TabTransformer만
프로젝트 규모에 맞게 확장한다.

| 기법 (수업) | LSTM — 수업과 동일하게 적용 | GBDT / TabTransformer — 확장 |
|-------------|------------------------------|------------------------------|
| **Dropout** | `Dropout(0.1)` (수업 값 그대로) | TT: `dropout ∈ {0.1,0.2,0.3}` 탐색 |
| **EarlyStopping** | `monitor='val_loss', patience=epoch*0.3` (수업 그대로) | TT: `monitor=val AUC(max)`; GBDT: `early_stopping_rounds` |
| **HPO** | scikeras `KerasClassifier` + **`GridSearchCV`** `{epochs, batch_size}` (수업 그대로) | `Optuna`(베이지안) |
| **train/val 분리** | `train_test_split(shuffle=False)` (수업 그대로) | fold 내 분리 |

> 회귀였던 수업과 달리 본 과제는 **이진 분류**이므로, 최소 변경만 적용한다:
> - `KerasRegressor` → `KerasClassifier`
> - 출력층 `Dense(1)` → `Dense(1, activation='sigmoid')`, loss `mse` → `binary_crossentropy`
> - `make_scorer(mean_squared_error)` → `make_scorer(roc_auc_score, needs_proba=True)`

---

## 5. 시퀀스 구성 전략 (LSTM 전용)

**슬라이딩 윈도우** — 각 샘플이 과거 정보만 포함하므로 시간 누수가 구조적으로 발생하지 않는다.

```python
def make_sliding_windows(df, group_keys, feature_cols, target_col, window_size=5):
    X_list, y_list = [], []
    for _, g in df.groupby(group_keys):
        g = g.sort_values('LapNumber')
        feats, tgts = g[feature_cols].values, g[target_col].values
        for i in range(window_size, len(g)):
            X_list.append(feats[i-window_size:i])   # 과거 N랩
            y_list.append(tgts[i])                   # 현재 랩 pit 여부
    return np.array(X_list), np.array(y_list)        # X:(N,5,8), y:(N,)

group_keys   = ['Driver','Race','Year']
seq_features = ['LapNumber','TyreLife','LapTime (s)','LapTime_Delta',
               'Cumulative_Degradation','RaceProgress','Position','Position_Change']
window_size  = 5   # 기본값. 3·10 으로 sensitivity analysis
```

- 근거: F1 pit 결정은 통상 3~5랩 전 타이어 열화 징후로 판단.
- window보다 짧은 레이스 그룹은 제외(또는 window 축소).

---

## 6. 실험 구성 (총 6개)

### Group 1 — 베이스라인 (수업 모델)

**실험 1: LSTM 단독** — 수업 노트북 구조를 그대로 사용(분류용 최소 변경만)
- 입력: `(samples, 5, 8)` 슬라이딩 윈도우
- 구조(수업과 동일): `LSTM(128, return_sequences=True) → LSTM(128) → Dropout(0.1) → Dense(1, sigmoid)`
- 컴파일: `loss=binary_crossentropy, optimizer=Adam(0.001), metrics=[AUC]`
- HPO(수업과 동일): `KerasClassifier` + `GridSearchCV(param_grid={'epochs':[20,40], 'batch_size':[16,32]})`
- 학습 종료(수업과 동일): `EarlyStopping(monitor='val_loss', patience=epoch*0.3)`,
  `train_test_split(shuffle=False)`로 val 확보
- window_size sensitivity(3/5/10) 포함

```python
# 수업 lstm_model() 을 분류용으로 최소 변경
def lstm_model():
    model = Sequential(name='PitStop_LSTM')
    model.add(LSTM(128, input_shape=(window_size, len(seq_features)), return_sequences=True))
    model.add(LSTM(128))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))           # 회귀 Dense(1) → 분류 sigmoid
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(0.001), metrics=[tf.keras.metrics.AUC()])
    return model
```

### Group 2 — Tabular 특화 모델

**실험 2: LightGBM 단독**
- 입력: row 단위 (윈도우 불필요), FE(interaction 5 + cat 조합 4)
- HPO: Optuna(`learning_rate, num_leaves, n_estimators, min_child_samples, subsample, colsample`)
- 출발점: `lr=0.05, num_leaves=127, n_estimators=1000, scale_pos_weight=4.03`

**실험 3: XGBoost 단독**
- 실험 2와 동일 피처셋·CV·FE, `device='cpu'` 강제, Optuna HPO

**실험 4: TabTransformer 단독 (신경망 필수 포함)**
- embed_dim=16~32, num_heads=4, transformer_blocks=2
- Pair-wise TargetEncoding, `Dropout` + `EarlyStopping(val AUC)`
- (선택) Transformer 임베딩 추출 → LightGBM/XGB 입력 변형 비교

### Group 3 — 앙상블

**실험 5: GBDT 앙상블** — LightGBM + XGBoost OOF 가중 평균(OOF score 비례 가중)

**실험 6: 전체 앙상블** — LGBM + XGB + TabTransformer OOF 가중 평균
- 시작 가중치 `[0.35, 0.30, 0.35]` → OOF correlation으로 diversity 측정 후 조정

---

## 7. HPO 상세

| 모델 | 탐색 방법 | 주요 탐색 공간 | 종료 조건 |
|------|-----------|----------------|-----------|
| LSTM | **GridSearchCV(scikeras)** — 수업 그대로 | `epochs{20,40}`, `batch_size{16,32}` | EarlyStopping `val_loss`, patience=epoch*0.3 |
| LightGBM | Optuna (50 trials) | lr, num_leaves, n_est, min_child, subsample, colsample | early_stopping_rounds 100 |
| XGBoost | Optuna (50 trials) | eta, max_depth, n_est, min_child_weight, subsample, colsample | early_stopping_rounds 100 |
| TabTransformer | Optuna (30 trials) | embed_dim, blocks, lr, dropout | EarlyStopping val AUC, patience 8 |

> LSTM의 탐색 공간(`epochs`, `batch_size`)·scorer 호출 방식은 수업 노트북과 동일하게 유지한다.
> GBDT/TabTransformer만 OOF AUC 기준 Optuna로 확장하며, best param으로 best epoch 만큼 전체 train+val 재학습한다.

---

## 8. 평가 항목

### 8-1. 주요 지표

| 실험 | 모델 | OOF AUC | Fold 분산 | 학습 시간 |
|------|------|---------|-----------|-----------|
| 1 | LSTM | - | - | - |
| 2 | LightGBM | - | - | - |
| 3 | XGBoost | - | - | - |
| 4 | TabTransformer | - | - | - |
| 5 | GBDT 앙상블 | - | - | - |
| 6 | 전체 앙상블 | - | - | - |

> 참고 현재 기록: TabTransformer 단독 Val AUC 0.9255, LSTM Val AUC 0.8740

### 8-2. 앙상블 diversity

```python
oof_df = pd.DataFrame({'lgbm':oof_lgbm, 'xgb':oof_xgb, 'tt':oof_tt})
print(oof_df.corr())                       # correlation matrix
gain = ensemble_auc - max(lgbm_auc, xgb_auc, tt_auc)
```

### 8-3. window_size sensitivity (LSTM)

| window_size | LSTM OOF AUC |
|-------------|--------------|
| 3 | - |
| 5 | - |
| 10 | - |

### 8-4. HPO 효과 (선택 보조)

| 모델 | 기본 파라미터 AUC | HPO 후 AUC | gain |
|------|-------------------|------------|------|

---

## 9. 가설

| 가설 | 근거 |
|------|------|
| GBDT > LSTM | 수치형 피처 우세, row 독립 처리가 본 데이터에 적합 |
| TabTransformer 단독 < LightGBM | 수치형 비중 높고 cat cardinality 낮음 |
| 전체 앙상블 > GBDT 앙상블 | TabTransformer의 cat interaction이 diversity 기여 |
| window_size=5 최적 | pit 징후가 3~5랩 전 나타나는 도메인 특성 |
| HPO/Dropout/EarlyStopping 적용 시 fold 분산 감소 | 과적합 억제 |

---

## 10. 실험 순서 및 일정 (안)

```
Day 1 : 슬라이딩 윈도우 파이프라인 + 공통 FE + GroupKFold split 고정
Day 2 : 실험 2, 3 (GBDT 베이스라인 + Optuna HPO)
Day 3 : 실험 1 (LSTM, GridSearchCV + window sensitivity)
Day 4 : 실험 4 (TabTransformer, GPU 권장)
Day 5 : 실험 5, 6 (앙상블 + OOF correlation 분석)
Day 6 : 결과 정리, 보고서 작성 (HPO 효과 표 포함)
```

---

## 11. 리스크 및 대응

| 리스크 | 대응 |
|--------|------|
| window보다 짧은 레이스 그룹 존재 | 해당 그룹 제외 또는 window 축소 |
| TabTransformer fold 분산 과대 | EarlyStopping patience↑, lr↓, Dropout↑ |
| GPU 없음 | Kaggle 노트북 T4 GPU 사용 |
| LSTM AUC 랜덤 수준 | window 조정 후 재실험, 한계로 명시 |
| 앙상블 gain 없음 | OOF correlation 높음을 원인 분석, 가중치 재조정 |
| TabTransformer 임베딩 추출 시 커널 크래시 | 배치 크기 축소 (이력 있음) |
| GridSearchCV가 분류 지표 미지원 | `make_scorer(roc_auc_score)`로 교체 |

---

## 12. 산출물

- 노트북: `LSTM.ipynb`, `TabTransformer.ipynb` (실험별 OOF 저장)
- 모델 가중치: `best_lstm.keras` (Keras), `best_tabtransformer.pth`
- 제출 파일: `submission_lstm.csv`, `submission_tabtransformer_ensemble.csv`
- 결과 표: 8-1 ~ 8-4 채운 최종 보고서
```
```
