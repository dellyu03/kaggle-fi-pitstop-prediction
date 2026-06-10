# Per-Fold OOF 루프 코드 설명

## 전체 구조

```
사전 준비 (루프 전)
  ↓
fold 0~4 반복
  ├─ ① 연속형 스케일링
  ├─ ② TabTransformer 학습
  ├─ ③ TT OOF/test 예측
  ├─ ④ 임베딩 추출
  ├─ ⑤ LightGBM 학습
  └─ ⑥ XGBoost 학습
  ↓
최종 OOF AUC 출력
```

---

## 사전 준비

### GroupKFold 설정

```python
gkf = GroupKFold(n_splits=5)
splits = list(gkf.split(train_df, y, groups=groups))
```

- `GroupKFold`: 같은 레이스+시즌(Race_Year)이 train/val에 섞이지 않도록 그룹 단위로 분리
- `groups = ['Monaco_2023', 'Monza_2022', ...]` 기준으로 104개 그룹을 5등분
- `list()`로 미리 저장 → TT/LGBM/XGB 세 모델이 동일한 splits 재사용

### OOF 저장 배열

```python
oof_tt  = np.zeros(len(train_df))  # 439,140개 0으로 초기화
oof_lgb = np.zeros(len(train_df))
oof_xgb = np.zeros(len(train_df))
```

- fold 루프에서 val 인덱스 위치에 예측값을 채워넣기 위한 빈 배열
- 5번의 fold가 끝나면 전체 439,140개가 채워짐

### test 예측 저장 리스트

```python
tt_test_folds, lgb_test_folds, xgb_test_folds = [], [], []
```

- fold마다 test 예측값을 append
- 루프 후 5개 fold 예측의 평균(fold bagging)을 최종 test 예측으로 사용

### GBDT 하이퍼파라미터

```python
lgb_params = {
    'objective': 'binary',      # 이진 분류
    'metric': 'auc',            # AUC 기준으로 평가
    'learning_rate': 0.05,      # 학습률
    'num_leaves': 127,          # 트리 복잡도
    'max_depth': -1,            # 깊이 제한 없음
    'min_child_samples': 50,    # 리프 최소 샘플 수
    'feature_fraction': 0.8,    # 피처 80%만 랜덤 사용
    'bagging_fraction': 0.8,    # 샘플 80%만 랜덤 사용
    'scale_pos_weight': spw,    # 클래스 불균형 보정 (4.026)
}

xgb_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 6,
    'tree_method': 'hist',      # GPU 히스토그램 방식
    'device': XGB_DEVICE,       # GPU 사용
    'scale_pos_weight': spw,
}
```

---

## fold 루프

### 루프 시작

```python
for f, (tr, va) in enumerate(splits):
    torch.cuda.reset_peak_memory_stats()  # GPU 메모리 측정 초기화
```

- `tr`: train 인덱스 배열 (약 351,312개)
- `va`: val 인덱스 배열 (약 87,828개)
- `enumerate`로 fold 번호(f)와 인덱스 쌍을 동시에 받음

---

### ① 연속형 per-fold 스케일링

```python
sc = StandardScaler().fit(train_df.iloc[tr][cont_features])
tr_df[cont_features] = sc.transform(tr_df[cont_features])
va_df[cont_features] = sc.transform(va_df[cont_features])
te_df[cont_features] = sc.transform(te_df[cont_features])
```

**핵심: train fold 데이터로만 fit**

| 데이터 | 처리 |
|--------|------|
| train fold | fit + transform |
| val fold | transform만 |
| test | transform만 |

val/test의 분포 정보가 scaler에 새지 않도록 누수 차단

---

### ② TabTransformer 학습

```python
model, tt_auc, ep_done = train_tabtransformer(
    tr_df, va_df, y[tr], y[va], cat_enc_cols, cont_features
)
torch.save(model.state_dict(), f'best_tabtransformer_fold{f}.pth')
peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6
```

- fold마다 TT를 처음부터 재학습 → 임베딩 누수 차단
- best val AUC 시점의 가중치를 파일로 저장
- GPU 피크 메모리 측정

**실제 결과:**

| Fold | TT AUC | 학습 시간 | Epoch |
|------|--------|-----------|-------|
| 0 | 0.9101 | 230s | 49 |
| 1 | 0.9166 | 269s | 58 |
| 2 | 0.9189 | 173s | 35 |
| 3 | 0.9100 | 122s | 25 |
| 4 | 0.8860 | 87s | 19 |

---

### ③ TT OOF / test 예측

```python
oof_tt[va] = predict_tt(model, va_df, cat_enc_cols, cont_features)
tt_test_folds.append(predict_tt(model, te_df, cat_enc_cols, cont_features))
```

- `oof_tt[va]`: val 인덱스 위치에 예측 확률 채워넣음
- `tt_test_folds`: fold별 test 예측 저장 (나중에 평균냄)

---

### ④ 임베딩 추출

```python
emb_tr = extract_embeddings(model, tr_df, cat_enc_cols, cont_features)
emb_va = extract_embeddings(model, va_df, cat_enc_cols, cont_features)
emb_te = extract_embeddings(model, te_df, cat_enc_cols, cont_features)
```

- TT의 마지막 예측값이 아닌 **73차원 중간 표현** 추출
- LGBM/XGB의 입력 피처로 사용
- TT가 학습한 범주형 피처 조합 관계가 담긴 벡터

```
raw 피처 → TT → 73차원 임베딩 → LGBM/XGB → 최종 예측
```

---

### ⑤ LightGBM 학습

```python
dtr = lgb.Dataset(emb_tr, label=y[tr])
dva = lgb.Dataset(emb_va, label=y[va])
lgbm = lgb.train(
    lgb_params, dtr,
    num_boost_round=1000,
    valid_sets=[dva],
    callbacks=[lgb.early_stopping(50, verbose=False)]  # 50번 연속 개선 없으면 종료
)
oof_lgb[va] = lgbm.predict(emb_va)
lgb_test_folds.append(lgbm.predict(emb_te))
```

- TT 임베딩(73차원)을 입력으로 학습
- Early Stopping: val AUC 기준 50 round
- OOF와 test 예측 저장

---

### ⑥ XGBoost 학습

```python
dxt = xgb.DMatrix(emb_tr, label=y[tr])
dxv = xgb.DMatrix(emb_va, label=y[va])
xgbm = xgb.train(
    xgb_params, dxt,
    num_boost_round=1000,
    evals=[(dxv, 'val')],
    early_stopping_rounds=50
)
oof_xgb[va] = xgbm.predict(dxv)
xgb_test_folds.append(xgbm.predict(xgb.DMatrix(emb_te)))
```

- LGBM과 동일한 구조, GPU(cuda) 사용
- `xgb.DMatrix`: XGBoost 전용 데이터 형식

---

### fold 마무리

```python
del model, emb_tr, emb_va, emb_te
gc.collect()
torch.cuda.empty_cache()
```

- 다음 fold를 위해 GPU/메모리 해제
- `del`: 파이썬 참조 제거
- `gc.collect()`: 가비지 컬렉터 강제 실행
- `empty_cache()`: GPU 캐시 반환

---

## 루프 후 최종 출력

```python
total_elapsed = time.time() - t0

# 전체 OOF AUC
roc_auc_score(y, oof_tt)   # 0.9097
roc_auc_score(y, oof_lgb)  # 0.9285
roc_auc_score(y, oof_xgb)  # 0.9297
```

**실제 결과:**

```
OOF AUC  TT 0.9097 | LGBM 0.9285 | XGB 0.9297
총 학습 시간: 967s (16.1분)
  TT 합계 881s | LGBM 합계 34s | XGB 합계 22s
```

---

## fold 루프 전체 흐름 요약

```
fold 0~4 반복:
  ① StandardScaler fit (train fold만)
  ② TT 재학습 → 가중치 저장
  ③ TT로 val/test 확률 예측 → oof_tt, tt_test_folds
  ④ TT 임베딩 추출 (73차원)
  ⑤ LGBM 학습 → oof_lgb, lgb_test_folds
  ⑥ XGB 학습  → oof_xgb, xgb_test_folds
  메모리 해제

5번 완료:
  oof_tt/lgb/xgb  → 전체 439,140개 채워짐 → OOF AUC 계산
  *_test_folds    → 앙상블용 fold별 test 예측 5개씩 저장
```
