---
marp: true
theme: default
paginate: true
backgroundColor: #0f1117
color: #e8eaf6
style: |
  section {
    font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
    padding: 40px 60px;
  }
  h1 { color: #e040fb; font-size: 2em; margin-bottom: 0.3em; }
  h2 { color: #7c4dff; font-size: 1.5em; border-bottom: 2px solid #7c4dff; padding-bottom: 0.2em; }
  h3 { color: #40c4ff; font-size: 1.1em; margin-top: 0.5em; }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85em;
  }
  th {
    background: #1a1a2e;
    color: #e040fb;
    padding: 8px 12px;
    text-align: center;
  }
  td {
    background: #16213e;
    color: #e8eaf6;
    padding: 6px 12px;
    text-align: center;
    border: 1px solid #333;
  }
  code {
    background: #1e1e2e;
    color: #a6e22e;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.9em;
  }
  pre {
    background: #1e1e2e;
    padding: 16px;
    border-radius: 8px;
    border-left: 4px solid #7c4dff;
  }
  .highlight { color: #ffeb3b; font-weight: bold; }
  .good { color: #69f0ae; font-weight: bold; }
  .bad { color: #ff5252; }
  .note {
    background: #1a1a2e;
    border-left: 4px solid #40c4ff;
    padding: 10px 16px;
    border-radius: 4px;
    font-size: 0.85em;
    margin-top: 1em;
  }
  section.title-slide {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
  }
  section.title-slide h1 {
    font-size: 2.4em;
    margin-bottom: 0.2em;
  }
  section.title-slide p {
    font-size: 1em;
    color: #b0bec5;
  }
---

<!-- _class: title-slide -->

# TabTransformer + GBDT 앙상블
## F1 피트스탑 예측 — CUDA 최적화 실험

**RTX 5060 Ti 16GB | GroupKFold 5-fold OOF**

`유승환` · 2026

---

## 목차

1. 문제 정의
2. 실험 설계 & 방법론
3. 모델 아키텍처
4. 실험 설정 (Hyperparameter)
5. 학습 파이프라인
6. 실험 결과
7. 앙상블 전략
8. Ablation Study — 임베딩 효용
9. 비교 실험 (exp01 vs exp02)
10. 결론 & 향후 과제

---

## 1. 문제 정의

### Task
F1 레이싱 데이터에서 **다음 랩에 피트스탑을 할지 예측** (Binary Classification)

| 항목 | 내용 |
|------|------|
| Train | 439,140 rows × 16 cols |
| Test  | 188,165 rows × 15 cols |
| Target | `PitNextLap` (0/1) |
| Positive Rate | **19.9%** (class imbalance 존재) |

### 피처 구성

| 종류 | 피처 |
|------|------|
| 범주형 (3) | `Driver`, `Compound`, `Race` |
| 연속형 (11) | `Year`, `LapNumber`, `Stint`, `TyreLife`, `Position`, `LapTime (s)`, `LapTime_Delta`, `Cumulative_Degradation`, `RaceProgress`, `Position_Change`, `PitStop` |

---

## 2. 실험 설계 & 방법론

### 핵심 원칙

- **per-fold TabTransformer 재학습** → 임베딩 누수 원천 차단
- **GroupKFold(Race + Year) 5-fold** → 동일 레이스 데이터 분리로 현실적 평가
- 기본 피처만 사용 (FE 미적용)
- 고정 하이퍼파라미터 (Optuna 미적용)

### CV 전략

```
groups = Race + Year  →  104개 고유 그룹
GroupKFold(n_splits=5)  →  5개 fold
```

<div class="note">
동일 레이스의 랩들이 같은 fold에 모이도록 분리 → 데이터 누수 방지
</div>

---

## 3. 모델 아키텍처 — TabTransformer

```
범주형 피처 (Driver, Compound, Race)
        ↓ Embedding (32 / 4 / 8 dim)
        ↓ Linear Projection → d_model=32
        ↓ Transformer Encoder (n_layers × n_heads)
        ↓ Flatten
연속형 피처 (11개) ──────────────────────────┐
        ↓ Concat (32×3 + 11 = 107 dim)       │
        └─────────────────────────────────────┘
        ↓ MLP (LayerNorm → 256 → 64 → 1)
        ↓ Sigmoid → P(PitNextLap)
```

### 임베딩 차원

| 피처 | 카테고리 수 | Embedding dim |
|------|------------|---------------|
| Driver | 887 | 32 |
| Compound | 5 | 4 |
| Race | 26 | 8 |

---

## 4. 실험 설정 — exp02

| 하이퍼파라미터 | 값 |
|---------------|-----|
| `n_layers` | **5** |
| `n_heads` | 4 |
| `dim_feedforward` | **512** |
| `dropout` | **0.08** |
| `n_epochs` | 100 |
| `patience` (Early Stopping) | 12 |
| `batch_size` | 4096 |
| `lr` | 2e-4 |
| `weight_decay` | 1e-4 |
| `warmup_epochs` | 5 |
| `grad_clip` | 1.0 |

### CUDA 최적화
- **AMP (Mixed Precision)** — FP16 연산으로 속도 향상
- `pin_memory=True` — CPU↔GPU 전송 가속
- XGBoost `device='cuda'` — GPU 트리 학습

---

## 5. 학습 파이프라인

```
For each fold in GroupKFold(5):
  ① per-fold StandardScaler (train fold만으로 fit)
  ② TabTransformer 재학습 (best val AUC → EarlyStopping)
  ③ TT로 OOF / Test 예측 확률 저장
  ④ TT 임베딩 추출 (train fold / val / test)
  ⑤ LightGBM 학습 (임베딩 피처)
  ⑥ XGBoost 학습 (임베딩 피처, GPU)
```

### 스케줄러

```
Warmup (5 epoch): LR 0.02 → 0.2  (linear)
Cosine Annealing (5~100 epoch): LR → 0
```

### Loss

- `BCEWithLogitsLoss(pos_weight=4.026)` — 클래스 불균형 보정

---

## 6. 실험 결과 — Fold별 AUC

| Fold | TT AUC | LGBM AUC | XGB AUC | TT 학습시간 | Epochs | GPU Peak |
|------|--------|----------|---------|------------|--------|----------|
| 0 | 0.8681 | 0.9234 | 0.9247 | 385s | 75 | 528MB |
| 1 | <span class="bad">0.6330</span> | **0.9279** | **0.9287** | 81s | 14 | 528MB |
| 2 | **0.8867** | 0.9237 | 0.9247 | 377s | 66 | 528MB |
| 3 | 0.8537 | 0.9185 | 0.9182 | 435s | 77 | 528MB |
| 4 | 0.8537 | 0.9080 | 0.9064 | 377s | 69 | 528MB |
| **OOF** | **0.8341** | **0.9207** | **0.9208** | **1656s** | **60.2** | — |

<div class="note">
Fold 1 TT AUC 0.6330 이상치 — 해당 fold의 Race/Year 구성이 다른 fold와 분포 상이할 가능성
</div>

---

## 6. 실험 결과 — 전체 성능 비교

| 모델 | OOF AUC | F1 @0.5 |
|------|---------|---------|
| TabTransformer 단독 | 0.8341 | — |
| LightGBM (TT 임베딩) | 0.9207 | — |
| XGBoost (TT 임베딩) | 0.9208 | — |
| GBDT 앙상블 (LGB+XGB) | 0.9215 | — |
| 전체 앙상블 (등가중) | 0.9099 | — |
| **전체 앙상블 (가중최적)** | **0.9215** | — |

### 총 학습 시간

| 모델 | 합계 |
|------|------|
| TabTransformer (×5 fold) | **1,656s (27.6분)** |
| LightGBM | 30s |
| XGBoost (GPU) | 16s |
| **전체** | **1,735s (28.9분)** |

---

## 7. 앙상블 전략

### OOF 상관관계 분석

|  | TT | LGB | XGB |
|--|-----|-----|-----|
| **TT** | 1.000 | 0.769 | 0.766 |
| **LGB** | 0.769 | 1.000 | **0.990** |
| **XGB** | 0.766 | 0.990 | 1.000 |

### Grid Search 최적 가중치

```python
best_w = (TT: 0.0,  LGB: 0.5,  XGB: 0.5)
최적 OOF AUC = 0.9215
앙상블 gain (vs 최고 단일) = +0.0008
```

<div class="note">
LGB–XGB 상관이 0.99로 매우 높아 앙상블 다양성 이득이 제한적. TT는 가중치 0으로 수렴 — TT 임베딩 품질 개선 여지 있음
</div>

---

## 8. Ablation Study — TT 임베딩 효용

### 질문: TT 임베딩이 실제로 도움이 되는가?

| 입력 피처 | 모델 | OOF AUC | 차이 |
|----------|------|---------|------|
| Raw 피처 (연속형 + 범주형 코드) | LightGBM | 0.9123 | — |
| **TT 임베딩 (107-dim)** | LightGBM | **0.9207** | **+0.0084** |

### 결론

```
TT 임베딩 활용  →  +0.0084 AUC 향상
범주형 피처의 컨텍스트 정보를 Transformer가 효과적으로 인코딩
```

<div class="note">
임베딩 효용이 +0.0084로 검증됨. 다만 TT 단독 AUC(0.8341)는 낮아 임베딩 특징 추출기로서의 역할이 더 적합
</div>

---

## 9. 비교 실험 — exp01 vs exp02

| 항목 | exp01 | exp02 |
|------|-------|-------|
| `n_layers` | 3 | **5** |
| `dim_feedforward` | 512 | 512 |
| `dropout` | 0.15 | **0.08** |
| `n_epochs` | 60 | **100** |
| **OOF TT AUC** | 0.7733 | **0.8341 (+0.0608)** |
| **OOF LGBM AUC** | **0.9273** | 0.9207 |
| **OOF 앙상블 AUC** | **0.9277** | 0.9215 |
| 학습 시간 (TT) | 697s | 1,656s |
| GPU Peak | 439MB | 528MB |

<div class="note">
레이어 증가 + dropout 감소로 TT AUC +0.06 향상. 하지만 LGBM/앙상블은 exp01이 소폭 우세 — TT의 임베딩 품질이 아직 LGBM 입력으로는 exp01이 더 좋은 특징 제공
</div>

---

## 10. 결론 & 향후 과제

### 주요 결과 요약

- TabTransformer를 **임베딩 추출기**로 활용 → LightGBM/XGBoost 입력
- TT 임베딩이 raw 피처 대비 **+0.0084 AUC 향상** 검증
- 최종 OOF AUC: **0.9215** (GBDT 앙상블)
- GPU (RTX 5060 Ti) 활용으로 **28.9분**에 5-fold 완료

### 향후 과제

| 방향 | 내용 |
|------|------|
| FE 적용 | 피처 엔지니어링으로 raw 피처 품질 향상 |
| Optuna 튜닝 | TT 하이퍼파라미터 최적화 |
| TT AUC 개선 | Fold 1 이상치 원인 분석 (0.6330) |
| 앙상블 다양성 | LGB–XGB 대신 더 다양한 모델 조합 |
| 모델 경량화 | 추론 속도 최적화 |

---

<!-- _class: title-slide -->

# Q & A

**실험명**: `exp02_layers5_ff512_ep100_dropout0.08`
**최종 OOF AUC**: 0.9215
**제출 파일**: `submission_tabtransformer_ensemble.csv`

---

## 부록 — 제출 파일 통계

```
saved submission_tabtransformer_ensemble.csv (188165, 2)

count   188165.000000
mean         0.298037
std          0.345593
min          0.002564
25%          0.020647
50%          0.093695
75%          0.635120
max          0.985445
```

최적 가중치: `TT × 0.0 + LGB × 0.5 + XGB × 0.5`
(5-fold 평균 test 예측 soft voting)
