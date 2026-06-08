# F1 Pit Stop Prediction - 프로젝트 계획서

## 프로젝트 개요

**대회**: Kaggle Playground Series S6E5
**목표**: F1 드라이버가 다음 랩에 피트스탑을 진행할지 예측 (이진 분류)
**타겟 변수**: `PitNextLap` (0 또는 1)

## 데이터 요약

| 구분 | 행 수 | 컬럼 수 |
|------|-------|---------|
| Train | 439,140 | 16 (타겟 포함) |
| Test | 188,165 | 15 |

### 피처 목록

| 피처 | 타입 | 설명 |
|------|------|------|
| Driver | 명목형 | 드라이버 식별자 (887개) |
| Compound | 명목형 | 타이어 종류 (HARD, MEDIUM, SOFT, INTERMEDIATE, WET) |
| Race | 명목형 | 그랑프리 이름 (26개) |
| Year | 순서형 | 연도 (2022-2025) |
| PitStop | 이진형 | 현재 랩 피트스탑 여부 |
| LapNumber | 연속형 | 현재 랩 번호 |
| Stint | 순서형 | 현재 스틴트 번호 |
| TyreLife | 연속형 | 현재 타이어 사용 랩 수 |
| Position | 순서형 | 현재 순위 |
| LapTime (s) | 연속형 | 랩타임 (초) |
| LapTime_Delta | 연속형 | 랩타임 변화량 |
| Cumulative_Degradation | 연속형 | 누적 타이어 성능 저하 |
| RaceProgress | 연속형 | 레이스 진행률 (0~1) |
| Position_Change | 연속형 | 순위 변동 |

---

## 실험 구조

두 가지 모델링 접근법을 비교하는 것이 핵심 목표입니다.

```
모델 A: TabTransformer + LightGBM + XGBoost (앙상블)
모델 B: LSTM (시계열 기반)
```

---

## Phase 1: 데이터 전처리 및 피처 엔지니어링

### 1-1. EDA 노트북 버그 수정
- `data_visualization.ipynb`의 `Variable Type` 컬럼명 참조 오류 수정
- 시각화 코드 정상 동작 확인

### 1-2. 공통 전처리 파이프라인
- 범주형 변수 인코딩 (Driver, Compound, Race)
- 수치형 변수 스케일링 (StandardScaler / MinMaxScaler)
- Train/Validation 분할 전략 수립 (StratifiedKFold 등, 클래스 불균형 고려)

### 1-3. 피처 엔지니어링
- 타이어 관련: TyreLife/Stint 비율, 타이어 잔여 수명 추정
- 레이스 컨텍스트: 남은 랩 수, 포지션 변화 추세
- LSTM용 시퀀스 데이터: 드라이버 x 레이스별 랩 시퀀스 구성

---

## Phase 2: 모델 A - TabTransformer + LightGBM + XGBoost

### 2-1. TabTransformer 임베딩 추출
- 범주형 변수(Driver, Compound, Race) → 임베딩 레이어
- Transformer 인코더로 범주형 임베딩 학습
- 학습된 임베딩을 피처로 추출

### 2-2. GBDT 앙상블
- TabTransformer 임베딩 + 수치형 피처 → LightGBM / XGBoost 입력
- 하이퍼파라미터 튜닝 (Optuna)
- 개별 모델 성능 및 앙상블(Soft Voting/Stacking) 비교

### 노트북
- `TabTransformer.ipynb`

---

## Phase 3: 모델 B - LSTM

### 3-1. 시퀀스 데이터 구성
- 드라이버별 랩 시퀀스를 시계열로 구성
- 고정 윈도우 크기로 입력 시퀀스 생성

### 3-2. LSTM 모델
- 단순 LSTM 아키텍처로 다음 랩 피트스탑 여부 예측
- 하이퍼파라미터 실험 (hidden size, num layers, dropout 등)

### 노트북
- `LSTM.ipynb` (신규 생성)

---

## Phase 4: 성능 비교 및 제출

### 평가 지표
- **AUC-ROC**: 주요 평가 지표
- **F1-Score**: 클래스 불균형 고려
- **Precision / Recall**: 피트스탑 예측 정확도

### 시각화
- 모델별 ROC 커브 비교
- 혼동 행렬 (Confusion Matrix)
- 피처 중요도 (GBDT 모델)

### Kaggle 제출
- `sample_submission.csv` 형식에 맞춘 예측 파일 생성

---

## Phase 5: 보고서 작성

### 보고서 구성
1. **문제 정의**: 대회 설명, 데이터 분석 요약
2. **모델 아키텍처**: TabTransformer, LightGBM, XGBoost, LSTM 각각 설명
3. **실험 설정**: 전처리, 하이퍼파라미터, 학습 환경
4. **결과 비교**: 성능 지표 비교표, 시각화
5. **결론 및 인사이트**: 어떤 접근법이 왜 더 효과적이었는지 분석

---

## 필요 패키지

| 패키지 | 용도 |
|--------|------|
| torch | TabTransformer, LSTM 구현 |
| lightgbm | GBDT 모델 |
| xgboost | GBDT 모델 |
| optuna | 하이퍼파라미터 튜닝 |
| scikit-learn | 전처리, 평가 (이미 설치됨) |
| pandas, seaborn, matplotlib | 데이터 처리, 시각화 (이미 설치됨) |

---

## 파일 구조

```
kaggle_f1-pit-stop_prediction/
├── data/
│   ├── kaggle_data/          # 원본 Kaggle 데이터
│   ├── edited_data/          # 가공 데이터
│   └── grandfrix_records/    # GP 기록 데이터
├── data_visualization.ipynb  # EDA 및 시각화
├── base.ipynb                # 베이스라인 모델
├── TabTransformer.ipynb      # Phase 2: TabTransformer + GBDT
├── LSTM.ipynb                # Phase 3: LSTM (신규)
├── PROJECT_PLAN.md           # 프로젝트 계획서 (본 문서)
└── REPORT.md                 # Phase 5: 최종 보고서 (예정)
```