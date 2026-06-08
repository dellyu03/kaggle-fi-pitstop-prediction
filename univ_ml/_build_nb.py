import json

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}

def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}

def _src(lines):
    text = "\n".join(lines)
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(
"# F1 Pit Stop 예측 — LSTM 실험 (범주형 임베딩 포함)",
"",
"과거 N랩 시퀀스로 다음 랩 피트 여부(`PitNextLap`)를 예측한다.",
"수업 LSTM(수치형 전용)을 확장하여 **범주형(Driver/Compound/Race)을 Embedding으로 시퀀스에 결합**한다.",
"",
"- 구조: `[연속형 + 범주형 Embedding] → LSTM(128) → LSTM(128) → Dropout(0.1) → Dense(1, sigmoid)`",
"- 범주형: 각 타임스텝의 코드를 Embedding → 시퀀스 피처로 concat (Keras **함수형 API**)",
"- HPO: `{epochs, batch_size}` grid search (GroupKFold + AUC)",
"- 학습 종료: `EarlyStopping(monitor='val_loss', patience=epoch*0.3)`",
"- 검증: **GroupKFold(Race+Year) 5-fold** (누수 방지)",
"- 평가: ROC-AUC (보조 F1 / confusion matrix)",
))

cells.append(md("## 0. import & 설정"))

cells.append(code(
"import time",
"from itertools import product",
"import numpy as np",
"import pandas as pd",
"import tensorflow as tf",
"from tensorflow.keras.models import Model",
"from tensorflow.keras.layers import Input, Embedding, Concatenate, LSTM, Dense, Dropout",
"from tensorflow.keras.optimizers import Adam",
"from tensorflow.keras.callbacks import EarlyStopping",
"from sklearn.preprocessing import LabelEncoder, StandardScaler",
"from sklearn.model_selection import GroupKFold",
"from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve",
"import matplotlib.pyplot as plt",
"",
"SEED = 42",
"np.random.seed(SEED)",
"tf.random.set_seed(SEED)",
"",
"# ===== 실험 설정 =====",
"WINDOW_SIZE  = 5         # 기본 윈도우 (sensitivity: 3 / 5 / 10)",
"USE_SCALER   = True      # 연속형 StandardScaler (train fold로만 fit)",
"POS_WEIGHT   = 4.03      # 클래스 불균형 대응 (neg/pos)",
"CLASS_WEIGHT = {0: 1.0, 1: POS_WEIGHT}",
"",
"num_features = ['LapNumber', 'TyreLife', 'LapTime (s)', 'LapTime_Delta',",
"               'Cumulative_Degradation', 'RaceProgress', 'Position', 'Position_Change']",
"cat_features = ['Driver', 'Compound', 'Race']      # 임베딩으로 결합",
"emb_dim_map  = {'Driver': 16, 'Compound': 3, 'Race': 6}",
"group_keys   = ['Driver', 'Race', 'Year']",
"target_col   = 'PitNextLap'",
"print('TF', tf.__version__)",
))

cells.append(md("## 1. 데이터 로드"))

cells.append(code(
"train = pd.read_csv('../data/kaggle_data/train.csv')",
"test  = pd.read_csv('../data/kaggle_data/test.csv')",
"print('train :', train.shape, '| test :', test.shape)",
"print('positive rate :', round(train[target_col].mean(), 4))",
"train[cat_features + num_features + [target_col]].head()",
))

cells.append(md(
"## 2. 데이터 정렬",
"",
"같은 (Driver, Race, Year) 시퀀스가 시간순(LapNumber)으로 이어지도록 **명시적으로 정렬**한다.",
"슬라이딩 윈도우가 과거→현재 순서를 전제로 하므로 이 단계가 선행되어야 한다.",
))

cells.append(code(
"train = train.sort_values(group_keys + ['LapNumber']).reset_index(drop=True)",
"test  = test.sort_values(group_keys + ['LapNumber']).reset_index(drop=True)",
"train[group_keys + ['LapNumber', target_col]].head(10)",
))

cells.append(md(
"## 3. 범주형 인코딩 (LabelEncoder)",
"",
"Driver/Compound/Race를 정수 코드로 변환. **train+test를 합쳐 fit**하여 test의 unseen 카테고리를 방지한다.",
"(라벨 번호만 부여하므로 누수 아님. 연속형 스케일링은 fold 내부에서 train으로만 fit.)",
))

cells.append(code(
"all_data = pd.concat([train, test], ignore_index=True)",
"cat_dims, emb_dims = [], []",
"for col in cat_features:",
"    le = LabelEncoder().fit(all_data[col])",
"    train[col + '_enc'] = le.transform(train[col])",
"    test[col + '_enc']  = le.transform(test[col])",
"    cat_dims.append(len(le.classes_))",
"    emb_dims.append(emb_dim_map[col])",
"",
"cat_enc_cols = [c + '_enc' for c in cat_features]",
"print('cat cardinality :', dict(zip(cat_features, cat_dims)))",
"print('emb dims         :', dict(zip(cat_features, emb_dims)))",
))

cells.append(md(
"## 4. 슬라이딩 윈도우 구성",
"",
"각 샘플은 **과거 window_size 랩**만 포함(시간 누수 없음).",
"연속형(`Xn`)과 범주형 코드(`Xc`)를 함께 만들고, GroupKFold용 `groups`(Race+Year)도 생성한다.",
))

cells.append(code(
"def make_sliding_windows(df, group_keys, num_cols, cat_cols, target_col, window_size=5):",
"    Xn, Xc, y, grp = [], [], [], []",
"    for keys, g in df.groupby(group_keys):",
"        g = g.sort_values('LapNumber')",
"        nums = g[num_cols].values",
"        cats = g[cat_cols].values            # 이미 라벨인코딩된 정수",
"        tgts = g[target_col].values",
"        race_year = f'{keys[1]}_{keys[2]}'",
"        for i in range(window_size, len(g)):",
"            Xn.append(nums[i-window_size:i])",
"            Xc.append(cats[i-window_size:i])",
"            y.append(tgts[i])",
"            grp.append(race_year)",
"    return (np.array(Xn, dtype='float32'),",
"            np.array(Xc, dtype='int32'),",
"            np.array(y,  dtype='float32'),",
"            np.array(grp))",
))

cells.append(code(
"Xn, Xc, y, groups = make_sliding_windows(",
"    train, group_keys, num_features, cat_enc_cols, target_col, WINDOW_SIZE)",
"print('Xn(연속형):', Xn.shape, '| Xc(범주형):', Xc.shape, '| y:', y.shape)",
"print('pos rate :', round(float(y.mean()), 4))",
))

cells.append(md(
"## 5. 데이터 분할 (GroupKFold)",
"",
"같은 레이스(Race+Year)는 한 fold에만 들어가도록 분할 → 같은 레이스의 랩이 train/val에 섞이는 누수 차단.",
"아래는 5-fold 중 첫 분할의 shape 확인.",
))

cells.append(code(
"gkf = GroupKFold(n_splits=5)",
"tr_idx, va_idx = next(gkf.split(Xn, y, groups=groups))",
"print('train 샘플 :', len(tr_idx), '| val 샘플 :', len(va_idx))",
"print('train 레이스 수 :', len(np.unique(groups[tr_idx])),",
"      '| val 레이스 수 :', len(np.unique(groups[va_idx])))",
"print('겹치는 레이스 :', len(set(groups[tr_idx]) & set(groups[va_idx])), '(0이어야 정상)')",
))

cells.append(md(
"## 6. 모델 정의 (범주형 임베딩 + LSTM, 함수형 API)",
"",
"연속형 시퀀스와 각 범주형의 Embedding 시퀀스를 마지막 축으로 concat → LSTM에 입력.",
))

cells.append(code(
"def to_inputs(Xn, Xc):",
"    \"\"\"모델 입력 형태로 변환: [연속형, cat0, cat1, ...]\"\"\"",
"    return [Xn] + [Xc[:, :, j] for j in range(Xc.shape[2])]",
"",
"def build_lstm(window_size, n_num, cat_dims, emb_dims, dropout=0.1):",
"    num_in = Input(shape=(window_size, n_num), name='num')",
"    cat_ins, embs = [], []",
"    for k, (card, ed) in enumerate(zip(cat_dims, emb_dims)):",
"        ci = Input(shape=(window_size,), name=f'cat_{cat_features[k]}', dtype='int32')",
"        e  = Embedding(input_dim=card, output_dim=ed)(ci)   # (window, ed)",
"        cat_ins.append(ci); embs.append(e)",
"    x = Concatenate(axis=-1)([num_in] + embs)               # (window, n_num+sum(ed))",
"    x = LSTM(128, return_sequences=True)(x)",
"    x = LSTM(128)(x)",
"    x = Dropout(dropout)(x)                                  # 수업: Dropout(0.1)",
"    out = Dense(1, activation='sigmoid')(x)                  # 분류 sigmoid",
"    model = Model([num_in] + cat_ins, out)",
"    model.compile(loss='binary_crossentropy',               # 분류 BCE",
"                  optimizer=Adam(learning_rate=0.001),",
"                  metrics=[tf.keras.metrics.AUC(name='auc')])",
"    return model",
"",
"build_lstm(WINDOW_SIZE, len(num_features), cat_dims, emb_dims).summary()",
))

cells.append(md(
"## 7. GroupKFold OOF 평가 함수",
"",
"연속형은 train fold로만 `StandardScaler` fit, 범주형 코드는 그대로 전달.",
))

cells.append(code(
"def run_oof(Xn, Xc, y, groups, window_size, epochs, batch_size,",
"            dropout=0.1, use_scaler=True, n_splits=5, verbose=0):",
"    gkf = GroupKFold(n_splits=n_splits)",
"    oof = np.zeros(len(y))",
"    fold_aucs = []",
"    for f, (tr, va) in enumerate(gkf.split(Xn, y, groups=groups)):",
"        Xn_tr, Xn_va = Xn[tr], Xn[va]",
"        if use_scaler:",
"            sc = StandardScaler()",
"            n, w, feat = Xn_tr.shape",
"            Xn_tr = sc.fit_transform(Xn_tr.reshape(-1, feat)).reshape(n, w, feat).astype('float32')",
"            Xn_va = sc.transform(Xn_va.reshape(-1, feat)).reshape(Xn_va.shape).astype('float32')",
"        m = build_lstm(window_size, Xn.shape[2], cat_dims, emb_dims, dropout)",
"        es = EarlyStopping(monitor='val_loss', mode='min',",
"                           patience=max(1, int(epochs * 0.3)),",
"                           restore_best_weights=True)",
"        m.fit(to_inputs(Xn_tr, Xc[tr]), y[tr],",
"              validation_data=(to_inputs(Xn_va, Xc[va]), y[va]),",
"              epochs=epochs, batch_size=batch_size,",
"              class_weight=CLASS_WEIGHT, callbacks=[es], verbose=verbose)",
"        p = m.predict(to_inputs(Xn_va, Xc[va]), verbose=0).ravel()",
"        oof[va] = p",
"        a = roc_auc_score(y[va], p)",
"        fold_aucs.append(a)",
"        print(f'  fold {f}  AUC = {a:.4f}')",
"    oof_auc = roc_auc_score(y, oof)",
"    print(f'>>> OOF AUC = {oof_auc:.4f} | mean {np.mean(fold_aucs):.4f} | std {np.std(fold_aucs):.4f}')",
"    return oof, fold_aucs",
))

cells.append(md(
"## 8. HPO — {epochs, batch_size} grid search",
"",
"표본 일부로 베스트 조합 탐색 (GroupKFold 2-fold + ROC-AUC).",
))

cells.append(code(
"param_grid = {'epochs': [20, 40], 'batch_size': [16, 32]}",
"",
"rng = np.random.default_rng(SEED)",
"samp = rng.choice(len(y), size=min(40000, len(y)), replace=False)",
"Xn_s, Xc_s, y_s, g_s = Xn[samp], Xc[samp], y[samp], groups[samp]",
"",
"results = []",
"for ep, bs in product(param_grid['epochs'], param_grid['batch_size']):",
"    print(f'[grid] epochs={ep}, batch_size={bs}')",
"    oof_s, _ = run_oof(Xn_s, Xc_s, y_s, g_s, WINDOW_SIZE, ep, bs,",
"                       use_scaler=USE_SCALER, n_splits=2, verbose=0)",
"    results.append({'epochs': ep, 'batch_size': bs, 'auc': round(roc_auc_score(y_s, oof_s), 4)})",
"",
"hpo_df = pd.DataFrame(results).sort_values('auc', ascending=False).reset_index(drop=True)",
"best_epochs = int(hpo_df.loc[0, 'epochs'])",
"best_bs     = int(hpo_df.loc[0, 'batch_size'])",
"print('best params :', {'epochs': best_epochs, 'batch_size': best_bs})",
"hpo_df",
))

cells.append(md("## 9. GroupKFold 5-fold OOF 평가 (최종 성능)"))

cells.append(code(
"t0 = time.time()",
"oof5, aucs5 = run_oof(Xn, Xc, y, groups, WINDOW_SIZE, best_epochs, best_bs,",
"                      dropout=0.1, use_scaler=USE_SCALER, n_splits=5, verbose=1)",
"print('elapsed (s):', round(time.time() - t0, 1))",
))

cells.append(md("## 10. window_size sensitivity (3 / 5 / 10)"))

cells.append(code(
"sens = {}",
"for ws in [3, 5, 10]:",
"    Xn_w, Xc_w, y_w, g_w = make_sliding_windows(",
"        train, group_keys, num_features, cat_enc_cols, target_col, ws)",
"    print(f'window_size={ws}  samples={len(y_w)}')",
"    oof_w, _ = run_oof(Xn_w, Xc_w, y_w, g_w, ws, best_epochs, best_bs,",
"                       use_scaler=USE_SCALER, n_splits=5, verbose=0)",
"    sens[ws] = round(roc_auc_score(y_w, oof_w), 4)",
"print('window_size sensitivity :', sens)",
))

cells.append(md("## 11. 평가 지표 (F1 / Confusion Matrix / ROC)"))

cells.append(code(
"pred_label = (oof5 >= 0.5).astype(int)",
"print('OOF AUC :', round(roc_auc_score(y, oof5), 4))",
"print('F1      :', round(f1_score(y, pred_label), 4))",
"print('Confusion Matrix:')",
"print(confusion_matrix(y, pred_label))",
"",
"fpr, tpr, _ = roc_curve(y, oof5)",
"plt.figure(figsize=(5, 5))",
"plt.plot(fpr, tpr, label=f'LSTM (AUC={roc_auc_score(y, oof5):.4f})')",
"plt.plot([0, 1], [0, 1], '--', color='gray')",
"plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('LSTM ROC Curve'); plt.legend()",
"plt.show()",
))

cells.append(md(
"## 12. 전체 재학습 → test 예측 → submission",
"",
"test의 각 행은 같은 그룹 내 직전 랩들로 윈도우를 만든다.",
"초반 랩(과거 < window_size)은 첫 랩으로 **left-pad**하여 모든 id에 예측을 부여한다.",
))

cells.append(code(
"def make_predict_windows(df, group_keys, num_cols, cat_cols, window_size):",
"    ids, Xn, Xc = [], [], []",
"    for keys, g in df.groupby(group_keys):",
"        g = g.sort_values('LapNumber')",
"        nums = g[num_cols].values",
"        cats = g[cat_cols].values",
"        idv  = g['id'].values",
"        for i in range(len(g)):",
"            sl = slice(max(0, i - window_size), i)",
"            wn, wc = nums[sl], cats[sl]",
"            if len(wn) < window_size:",
"                pad = window_size - len(wn)",
"                wn = np.vstack([np.repeat(nums[0:1], pad, axis=0), wn]) if len(wn) else np.repeat(nums[0:1], window_size, axis=0)",
"                wc = np.vstack([np.repeat(cats[0:1], pad, axis=0), wc]) if len(wc) else np.repeat(cats[0:1], window_size, axis=0)",
"            ids.append(idv[i]); Xn.append(wn); Xc.append(wc)",
"    return np.array(ids), np.array(Xn, 'float32'), np.array(Xc, 'int32')",
))

cells.append(code(
"# 전체 train으로 최종 모델 재학습",
"scaler, Xn_all = None, Xn",
"if USE_SCALER:",
"    scaler = StandardScaler()",
"    n, w, feat = Xn.shape",
"    Xn_all = scaler.fit_transform(Xn.reshape(-1, feat)).reshape(Xn.shape).astype('float32')",
"",
"final = build_lstm(WINDOW_SIZE, Xn.shape[2], cat_dims, emb_dims, 0.1)",
"final.fit(to_inputs(Xn_all, Xc), y, epochs=best_epochs, batch_size=best_bs,",
"          class_weight=CLASS_WEIGHT, verbose=1)",
"final.save('best_lstm.keras')",
"print('saved best_lstm.keras')",
))

cells.append(code(
"ids_test, Xn_test, Xc_test = make_predict_windows(",
"    test, group_keys, num_features, cat_enc_cols, WINDOW_SIZE)",
"if scaler is not None:",
"    nt, wt, ft = Xn_test.shape",
"    Xn_test = scaler.transform(Xn_test.reshape(-1, ft)).reshape(Xn_test.shape).astype('float32')",
"",
"proba = final.predict(to_inputs(Xn_test, Xc_test), verbose=1).ravel()",
"sub = pd.DataFrame({'id': ids_test, 'PitNextLap': proba})",
"sub = sub.set_index('id').loc[test['id']].reset_index()   # sample 순서 정렬",
"sub.to_csv('submission_lstm.csv', index=False)",
"print(sub.shape); sub.head()",
))

cells.append(md(
"## 13. 결과 기록",
"",
"| 항목 | 값 |",
"|------|-----|",
"| OOF AUC (window=5) | |",
"| Fold std | |",
"| F1 (thr=0.5) | |",
"| window sensitivity (3/5/10) | |",
"| best {epochs, batch_size} | |",
))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("C289039_유승환_F1_LSTM.ipynb", "w") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("notebook written:", len(cells), "cells")
