# F1 Pit Stop 예측 실험 계획서 
1. 실험 목적
Kaggle Playground Series S6E5 (F1 Pit Stop 예측) 데이터셋을 기반으로, 수업에서 다룬 딥러닝 모델(LSTM)과 tabular 특화 모델(LightGBM, XGBoost, TabTransformer) 및 앙상블 전략을 체계적으로 비교해 각 모델 유형의 성능 차이를 ROC-AUC 기준으로 정량 측정한다.

2. 데이터셋 요약
항목내용출처Kaggle PS S6E5, 원본 F1 Strategy Dataset학습 데이터439,140행 × 14 피처타겟PitNextLap (이진 분류)평가지표ROC-AUC결측치없음클래스 불균형StratifiedKFold로 대응

3. 시퀀스 구성 전략 (LSTM)
슬라이딩 윈도우 방식 채택
LSTM은 과거 N랩을 보고 현재 랩의 pit 여부를 예측하는 구조로 설계한다. 슬라이딩 윈도우를 사용하면 각 샘플이 과거 정보만 포함하므로 데이터 누수가 구조적으로 발생하지 않는다.
window_size = 5 기준 예시 (D109, Canadian GP)

[랩1, 랩2, 랩3, 랩4, 랩5] → 랩6 PitNextLap 예측
[랩2, 랩3, 랩4, 랩5, 랩6] → 랩7 PitNextLap 예측
[랩3, 랩4, 랩5, 랩6, 랩7] → 랩8 PitNextLap 예측
...
pythondef make_sliding_windows(df, group_keys, feature_cols,
                         target_col, window_size=5):
    X_list, y_list = [], []
    for _, group in df.groupby(group_keys):
        group = group.sort_values('LapNumber')
        features = group[feature_cols].values
        targets  = group[target_col].values
        for i in range(window_size, len(group)):
            X_list.append(features[i-window_size:i])
            y_list.append(targets[i])
    return np.array(X_list), np.array(y_list)

# 결과 shape
# X: (총 샘플 수, 5, 8)  ← (samples, timesteps, features)
# y: (총 샘플 수,)
window_size 설정 기준
기본값: window_size = 5
근거:   F1 pit 결정은 통상 3~5랩 전 타이어 열화 징후로 판단
추가:   window_size = 3, 10 sensitivity analysis 수행
시퀀스 구성에 사용할 피처
pythonseq_features = [
    'LapNumber', 'TyreLife', 'LapTime (s)',
    'LapTime_Delta', 'Cumulative_Degradation',
    'RaceProgress', 'Position', 'Position_Change'
]
group_keys = ['Driver', 'Race', 'Year']

4. 실험 구성
총 6개 실험을 단계적으로 진행한다.
Group 1 — 베이스라인 (수업 모델)
실험 1: LSTM 단독
슬라이딩 윈도우로 구성한 시퀀스를 입력으로 받아 현재 랩의 pit 여부를 예측한다.
입력: (samples, 5, 8)
구조: Masking → LSTM(128) → LSTM(64) → Dense(1, sigmoid)
CV:   StratifiedKFold 5-fold

Group 2 — Tabular 특화 모델
실험 2: LightGBM 단독

입력: row 단위 (슬라이딩 윈도우 불필요)
피처 엔지니어링: arithmetic interaction 5개, cat 조합 4개
5-fold StratifiedKFold OOF
하이퍼파라미터: learning_rate=0.05, num_leaves=127, n_estimators=1000

실험 3: XGBoost 단독

실험 2와 동일한 피처셋 및 CV 전략 적용

실험 4: TabTransformer 단독 (신경망 필수 포함)

Keras 구현, embed_dim=16, num_heads=4, transformer_blocks=2
Pair-wise TargetEncoding (cuML GPU)
5-fold StratifiedKFold OOF


Group 3 — 앙상블
실험 5: GBDT 앙상블

LightGBM + XGBoost OOF 가중 평균
가중치: OOF score 비례로 설정

실험 6: 전체 앙상블

LightGBM + XGBoost + TabTransformer OOF 가중 평균
시작 가중치: [0.35, 0.30, 0.35]
OOF correlation matrix로 diversity 측정 후 조정


5. 공통 실험 조건
CV 전략:     StratifiedKFold (n_splits=5, shuffle=True, random_state=42)
             (전 실험 동일 적용 — 공정 비교 보장)
평가지표:    ROC-AUC (OOF 기준)
피처셋:      전 모델 동일한 피처 사용
시드:        42 고정
외부 데이터: 원본 F1 데이터셋 concat 허용 (전 실험 동일 적용)

6. 평가 항목
6-1. 주요 지표
실험모델OOF AUCFold 분산학습 시간1LSTM---2LightGBM---3XGBoost---4TabTransformer---5GBDT 앙상블---6전체 앙상블---
6-2. 앙상블 분석 지표
python# OOF correlation matrix — diversity 측정
oof_df = pd.DataFrame({
    'lgbm': oof_lgbm,
    'xgb':  oof_xgb,
    'tt':   oof_tt
})
print(oof_df.corr())

# 앙상블 gain
gain = ensemble_auc - max(lgbm_auc, xgb_auc, tt_auc)
6-3. window_size sensitivity analysis
window_sizeLSTM OOF AUC3-5-10-

7. 예상 결과 및 가설
가설근거GBDT > LSTM수치형 피처 7개 우세, row 독립 처리가 이 데이터에 적합TabTransformer 단독 < LightGBM수치형 비중 높고 cat cardinality 낮음전체 앙상블 > GBDT 앙상블TabTransformer의 cat interaction이 diversity 기여window_size=5 최적pit 징후가 3~5랩 전 나타나는 도메인 특성

8. 실험 순서 및 일정 (안)
Day 1:   슬라이딩 윈도우 파이프라인 구성 + 공통 피처 엔지니어링
Day 2:   실험 2, 3 (GBDT 베이스라인 확보)
Day 3:   실험 1 (LSTM, window_size sensitivity 포함)
Day 4:   실험 4 (TabTransformer, GPU 필요)
Day 5:   실험 5, 6 (앙상블, OOF correlation 분석)
Day 6:   결과 정리, 보고서 작성

9. 리스크 및 대응
리스크대응window_size보다 짧은 레이스 그룹 존재해당 그룹 제외 또는 window_size 축소TabTransformer fold 분산 과대EarlyStopping patience 늘리기, lr 낮추기GPU 없을 시 TabTransformer 학습 불가Kaggle 노트북 T4 GPU 사용LSTM AUC가 랜덤 수준일 경우window_size 조정 후 재실험, 한계로 명시앙상블 gain 없을 경우OOF correlation 높음을 원인으로 분석, 가중치 재조정