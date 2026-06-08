# TabTransformer + GBDT 앙상블 실험 계획서 — F1 Pit Stop 예측

> Kaggle Playground Series S6E5 · 타겟 `PitNextLap` (이진 분류) · 평가지표 ROC-AUC
> 행(랩) 단위 tabular 데이터에서 **다음 랩 피트 여부**를 예측한다.
> TabTransformer로 학습한 **피처 임베딩을 GBDT(LightGBM/XGBoost)의 입력**으로 쓰고, 세 모델을 **Soft Voting 앙상블**한다.

---

## 0. 실험 목적

1. TabTransformer로 학습한 임베딩이 raw 피처 대비 GBDT 성능을 끌어올리는가? (임베딩 효용, Ablation)
2. TT + LightGBM + XGBoost **Soft Voting 앙상블**의 gain은? 모델 간 예측 다양성(diversity)은 충분한가?
3. 등가중 vs OOF 기반 **가중 최적화** 앙상블 중 무엇이 우수한가?

---

## 1. 데이터 & 피처

| 구분 | 피처 | 처리 |
|------|------|------|
| 연속형(11) | `Year`, `LapNumber`, `Stint`, `TyreLife`, `Position`, `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation`, `RaceProgress`, `Position_Change`, `PitStop` | StandardScaler |
| 범주형(3) | `Driver`(≈887), `Compound`(5), `Race`(26) | LabelEncoder → Embedding |
| 타깃 | `PitNextLap` (0/1, pos rate ≈ 0.199) | `pos_weight` / `scale_pos_weight` |
| 제외 | `id` | — |

> `PitStop`(현재 랩 피트 여부)은 test에도 존재 → **추론 시점에 알 수 있는 합법 피처**.
> 단 타깃(다음 랩)과 의미가 가까우므로 누수 여부를 Step1에서 반드시 점검한다.

---

## 2. 전체 파이프라인 (스텝 요약)

```
Step1 EDA·누수점검 → Step2 전처리(인코딩+스케일링)
   → Step3 TabTransformer 학습 → Step4 임베딩 추출(누수 차단)
   → Step5 LightGBM(OOF) → Step6 XGBoost(OOF)
   → Step7 Soft Voting 앙상블(가중 최적화) → Step8 결과·Ablation
   → Step9 Test 추론 → 제출
```

---

## Step 1 — 피처 분석 (EDA) & 누수 점검

- **의도**: 피처 타입 확정 + 예측력 파악 + **누수 점검** + 임베딩 차원 근거 마련. `data_visualization.ipynb`의 `resumetable()` 방식을 확장한다.
- **기술**:
  - 피처 요약표: 각 피처의 `dtype / 결측수 / 고윳값수 / 예시 / 변수타입(이진·명목·순서·연속)` 산출.
  - 타깃 분포: `value_counts(normalize=True)` → pos rate ≈ 0.199 → `pos_weight ≈ 4.03`.
  - 범주형 예측력: `groupby(col)['PitNextLap'].mean()` (Compound·Race 변별력 확인).
  - 연속형 예측력: `sns.kdeplot(x=col, hue='PitNextLap')` (TyreLife·Degradation 가설 검증).
  - 상관/다중공선성: `corr().heatmap` → 고상관 쌍은 제거 대신 상호작용 피처로 활용.
- **누수 점검(가장 중요)**:
  - `PitStop` → `groupby('PitStop')['PitNextLap'].mean()` 으로 "방금 피트 시 다음 랩 피트 급감"이 자연스러운지 확인 → 합법 판정.
  - **단일 피처 AUC**: 각 피처 단독 `roc_auc_score(y, feature)` 가 0.99+ 면 누수 의심 → 정밀 조사 후 제외 결정.
- **세부**: 임베딩 차원 근거 — Driver=32(고cardinality), Race=8(중간), Compound=4(저). 경험칙 `min(50,(card+1)//2)` 대비 타당성 확인.

---

## Step 2 — 전처리 (인코딩 + 스케일링 + FE)

- **의도**: 범주형을 임베딩 입력용 정수 코드로, 연속형을 스케일 정규화로 변환하고, EDA 근거로 상호작용 피처를 생성한다.
- **기술**:
  - 범주형 `LabelEncoder`: **train+test 합쳐 fit** → test unseen 카테고리 방지(라벨 번호만 부여하므로 누수 아님).
  - 연속형 `StandardScaler`: **train(또는 train fold)으로만 fit** 후 transform → 검증 누수 차단.
  - **FE(상호작용 피처)는 이번 실험에서 적용하지 않음** — 기본 피처만 사용해 "임베딩+GBDT"와 "raw+GBDT" 베이스라인을 깨끗하게 비교(임베딩 효용 측정)한다. 향후 확장 항목으로 남김.
- **주의**: LabelEncoder만 train+test 합쳐 fit 허용(라벨 부여라 누수 아님). StandardScaler는 train fold로만 fit. (확장 시) TargetEncoding도 반드시 train fold로만 fit.

---

## Step 3 — TabTransformer 학습 (임베딩 사전학습)

- **의도**: 범주형을 밀집 임베딩으로 학습하고 Transformer로 피처 간 상호작용을 인코딩해, GBDT에 넘길 **표현력 높은 임베딩**을 사전학습한다.
- **기술**: PyTorch. 범주형 임베딩 → Linear projection으로 차원 통일 → `nn.TransformerEncoder` → 연속형 concat → MLP head → logit.
  | 항목 | 설정 |
  |------|------|
  | 임베딩 | Driver=32, Compound=4, Race=8 → **Linear projection 32차원 통일** |
  | Encoder | `nn.TransformerEncoder` (n_heads=4, n_layers=2, d_ff=256) |
  | 헤드 | Transformer 출력 + 연속형 concat → MLP → logit |
  | Loss | `BCEWithLogitsLoss(pos_weight≈4.03)` (불균형 대응) |
  | Optimizer | `AdamW(lr=5e-4, weight_decay=1e-4)` |
  | Scheduler | `CosineAnnealingLR(T_max=20)` |
  | Epoch / ES | 20 epoch, **EarlyStopping(val AUC, patience=5)**, best 저장 |
  | 저장 | `best_tabtransformer.pth` |
- **세부**: 대용량 배치 임베딩 추출 시 커널 크래시 이력 → `batch_size=4096` 유지, 필요 시 축소.

---

## Step 4 — 임베딩 추출 + 누수 차단

- **의도**: TT의 Transformer 출력과 연속형을 concat한 **107차원 임베딩**을 GBDT 입력으로 추출하되, OOF AUC가 부풀려지지 않도록 누수를 차단한다.
- **핵심 이슈**: 전체 train으로 TT 학습 → 그 임베딩을 GBDT의 val에 쓰면, val 행이 TT 학습 중 이미 노출되어 **OOF가 낙관적으로 편향**된다.
- **기술**:
  | 방식 | 절차 | 장단점 |
  |------|------|--------|
  | **(A) 엄밀 — per-fold (권장)** | fold마다 train부분으로 TT 재학습 → val부분 임베딩 추출 | 누수 없음 / TT 5회 학습으로 느림 |
  | (B) 간이 | 전체 train으로 TT 1회 학습 → 전체 임베딩 | 빠름 / OOF 낙관 편향(보고서 한계 명시) |
- **결정: (A) per-fold TT 재학습 채택** — 누수 없는 OOF 측정이 목적이므로 fold마다 TT를 재학습한다(TT 5회 학습 비용 감수). 이때 GBDT와 **동일한 GroupKFold 분할**을 공유한다. test 임베딩은 **전체 train으로 학습한 TT**(또는 fold 평균)로 추출.

---

## Step 5 — LightGBM (임베딩 입력, 5-fold OOF)

- **의도**: TT 임베딩을 입력으로 GBDT가 비선형 결정경계를 학습. 누수 없는 OOF 예측을 확보해 앙상블·비교의 기준으로 삼는다.
- **기술**: `LightGBM`, 5-fold OOF 루프(각 fold val 예측 누적 → `oof_lgbm`).
```python
lgb_params = {
    'objective': 'binary', 'metric': 'auc',
    'learning_rate': 0.05, 'num_leaves': 127, 'max_depth': -1,
    'min_child_samples': 50, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 1,
    'scale_pos_weight': (1-pos_rate)/pos_rate,  # ≈4.03 불균형 대응
    'n_jobs': -1, 'verbose': -1, 'random_state': 42,
}
# num_boost_round=1000, early_stopping(50)
```
- **세부**: 하이퍼파라미터는 **위 고정값 사용**(Optuna 튜닝 미적용). 앙상블 gain·임베딩 효용 검증에 집중하기 위함이며, Optuna는 향후 확장 항목으로 남김.

---

## Step 6 — XGBoost (임베딩 입력, 5-fold OOF)

- **의도**: LightGBM과 다른 부스팅 구현으로 **모델 다양성**을 확보 → 앙상블 gain의 원천을 만든다. 동일 fold split을 써서 공정 비교.
- **기술**: `XGBoost`, LightGBM과 같은 5-fold split 사용(`oof_xgb` 누적).
```python
xgb_params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 10,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'scale_pos_weight': (1-pos_rate)/pos_rate,
    'tree_method': 'hist', 'device': 'cpu',   # XGBoost MPS 미지원 → cpu 강제
    'n_jobs': -1, 'random_state': 42,
}
```
- **세부**: MPS 미지원으로 `device='cpu'` 고정.

---

## Step 7 — 앙상블 (Soft Voting + 가중 최적화)

- **의도**: 상관이 낮은 세 모델의 확률을 결합해 단일 모델보다 안정적이고 높은 AUC를 얻는다. 등가중 대비 OOF 기반 가중이 우수한지 검증.
- **기술**:
```python
# 1) 베이스라인: 등가중
ens_equal = (oof_tt + oof_lgbm + oof_xgb) / 3

# 2) diversity 측정 (상관 낮을수록 앙상블 이득↑)
oof_df = pd.DataFrame({'tt': oof_tt, 'lgbm': oof_lgbm, 'xgb': oof_xgb})
print(oof_df.corr())

# 3) 가중 최적화: OOF AUC 최대화 (w 합=1 제약, grid search 또는 scipy.optimize)
best_w = argmax_w  roc_auc_score(y, w_tt*oof_tt + w_lgbm*oof_lgbm + w_xgb*oof_xgb)
```
- **세부**: 시작 가중치 `[0.35(tt), 0.35(lgbm), 0.30(xgb)]` → OOF 기준 조정. **GBDT 앙상블(LGBM+XGB)** vs **전체 앙상블(+TT)** 둘 다 측정해 **TT 기여분을 분리**한다.

---

## Step 8 — 결과 확인 / 시각화 / Ablation

- **의도**: 단일 AUC 외에 앙상블 gain, diversity, 임베딩 효용까지 정량화해 결론의 근거를 만든다.
- **기술**:
  - 모델별 성능표(아래), 앙상블 gain = `ensemble_auc - max(개별 auc)`(양수여야 의미), OOF corr(0.95+ 면 gain 미미).
  - 시각화 3종: ROC Curve(4개 모델 겹침), Confusion Matrix(최종 앙상블 thr=0.5), 클래스별 예측 확률 분포.
  - Ablation: raw 피처+LGBM vs TT임베딩+LGBM → 차이 = 임베딩 효용.
- **모델별 성능표(보고서용)**:

| 모델 | OOF AUC | Fold std | F1(thr=0.5) | 비고 |
|------|---------|----------|-------------|------|
| TabTransformer 단독 | | | | |
| LightGBM (임베딩) | | | | |
| XGBoost (임베딩) | | | | |
| GBDT 앙상블 (LGBM+XGB) | | | | |
| 전체 앙상블 (등가중) | | | | |
| **전체 앙상블 (가중최적)** | | | | 최종 제출 후보 |

- **Ablation 표**:

| 구성 | OOF AUC | 해석 |
|------|---------|------|
| raw 피처 + LightGBM | | 임베딩 미사용 베이스 |
| TT 임베딩 + LightGBM | | 임베딩 효용 = (이 값 − 위 값) |

> 임베딩이 raw 대비 향상이 없으면 "TabTransformer 임베딩의 부가가치 제한적"으로 정직하게 결론.

---

## Step 9 — Test 추론 → 제출

- **의도**: CV로 검증한 설정으로 test 전 행에 예측을 부여하고 제출 파일을 생성한다.
- **기술**:
  - 전체 train으로 TT 재학습(best epoch) → test 임베딩 추출.
  - 5-fold로 학습한 LGBM/XGB 모델들의 test 예측 **평균(fold bagging)**.
  - Step7 최적 가중치로 soft voting → `id, PitNextLap`(확률)로 저장.
```python
final = w_tt*tt_test + w_lgbm*lgbm_test + w_xgb*xgb_test
submission['PitNextLap'] = final
submission.to_csv('submission_tabtransformer_ensemble.csv', index=False)
```

---

## 3. 검증 전략 (공통)

```
CV       : GroupKFold(n_splits=5), groups = Race+Year
           (LSTM 실험과 동일 CV로 통일 → 같은 표에서 공정 비교)
평가지표 : ROC-AUC (OOF 기준), 보조 F1 / Confusion Matrix
시드     : 42 고정
```

> **LSTM과 CV 통일**: 두 모델을 같은 기준으로 비교하기 위해 GroupKFold(Race+Year)를 사용한다.
> 같은 레이스의 랩이 train/val에 동시에 들어가는 누수도 함께 차단된다.
> TabTransformer(per-fold 재학습)·LGBM·XGB가 **모두 동일한 `groups` 분할**을 공유해야 OOF·앙상블이 정합적이다.

---

## 4. 가설

| 가설 | 근거 |
|------|------|
| TT 임베딩 + GBDT > raw + GBDT | Transformer가 피처 상호작용·드라이버 성향을 임베딩에 압축 |
| 앙상블 gain > 0 | LGBM/XGB/TT의 결정경계·표현이 달라 예측 다양성 존재 |
| 가중 최적 ≥ 등가중 | 모델별 OOF AUC 차이를 반영하면 더 우수 |
| `PitStop`은 합법 피처 | test 존재 + "방금 피트 시 다음 랩 피트 급감"이 도메인상 자연스러움 |
| 핵심 변별 피처: TyreLife·Degradation·Stint·RaceProgress | 타이어 열화·전략 타이밍이 피트 결정의 직접 원인 |

---

## 5. 산출물

- 노트북: `TabTransformer.ipynb` (본 계획 반영: fold0 → 5-fold OOF · 가중 앙상블)
- 모델: `best_tabtransformer.pth` (+ fold별 LGBM/XGB 모델)
- 제출: `submission_tabtransformer_ensemble.csv`
- 결과: Step8 표/그래프 + 모델 비교표 채운 보고서

---

## 6. 리스크 & 대응

| 리스크 | 대응 |
|--------|------|
| 임베딩 누수로 OOF 낙관 | per-fold TT 재학습(Step4-A), 불가 시 한계 명시 |
| 임베딩 추출 커널 크래시 | batch_size 축소(4096→2048), 청크 처리 |
| 앙상블 gain ≈ 0 | OOF corr 높음(0.95+) 원인 분석, 가중치 재조정·모델 다양화 |
| XGBoost MPS 미지원 | `device='cpu'` 강제(설정 완료) |
| 임베딩 효용 없음 | raw+GBDT가 더 나으면 그 결과를 정직하게 보고 |
| TT fold 분산 과대 | lr↓, weight_decay↑, patience↑ |
| `PitStop` 누수 의심 | Step1 단일 피처 AUC로 점검 후 제외 여부 결정 |
| 스케일링/타깃인코딩 누수 | scaler·target encoder는 train fold로만 fit |
