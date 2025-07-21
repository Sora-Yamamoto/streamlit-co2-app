import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import math

# パラメータ
ROLLING_WINDOW   = 10
STD_THRESHOLD    = 0.005
PVALUE_THRESHOLD = 0.05
DELAY            = 200  # 秒

def extract_stable_median(df, time_col, ar_col, start, end):
    # 指定区間データを抽出
    seg = df[(df[time_col] >= start + DELAY) & (df[time_col] <= end)].reset_index(drop=True)
    if seg.empty:
        return None, [f"範囲にデータがありませんでした ({start}s–{end}s)"], None, None

    # データ点数がウィンドウ未満なら全区間中央値
    if len(seg) < ROLLING_WINDOW:
        med_all = seg[ar_col].median()
        logs = [f"データ点数<{ROLLING_WINDOW} → 区間全体中央値 {med_all:.6f} を使用"]
        t0  = float(seg[time_col].iloc[0])
        dur = float(seg[time_col].iloc[-1] - seg[time_col].iloc[0])
        return med_all, logs, t0, dur

    # サンプリング間隔
    dt = seg[time_col].diff().dropna()
    sample_int = dt.mode().iloc[0] if not dt.empty else 1

    # 移動標準偏差
    seg['rolling_std'] = seg[ar_col].rolling(window=ROLLING_WINDOW).std()
    stable_idxs = seg.index[seg['rolling_std'] < STD_THRESHOLD].tolist()

    logs = []
    best_pval  = 1.0
    best_median= None
    best_time  = None
    best_dur   = None

    # 通常候補探索
    for idx in stable_idxs:
        if idx + ROLLING_WINDOW > len(seg):
            continue
        window = seg[ar_col].iloc[idx:idx+ROLLING_WINDOW]
        if window.nunique() == 1:
            continue
        stat, pval = adfuller(window)[:2]
        t0 = float(seg[time_col].iloc[idx])
        # 継続時間
        run, k = 0, idx
        while k in stable_idxs:
            run += 1; k += 1
        dur = run * sample_int
        med = window.median()
        status = '定常' if pval < PVALUE_THRESHOLD else '非定常'
        logs.append(f"Time {t0:.1f}s | ADF={stat:.3f}, p={pval:.3f}, {status}, 継続{dur:.0f}s, 中央値={med:.6f}")
        if pval < PVALUE_THRESHOLD:
            return med, logs, t0, dur
        if pval < best_pval:
            best_pval   = pval
            best_median = med
            best_time   = t0
            best_dur    = dur

    # フォールバック：最小stdウィンドウを使用
    idx_min = int(seg['rolling_std'].idxmin())
    window = seg[ar_col].iloc[idx_min:idx_min+ROLLING_WINDOW]
    med     = window.median()
    t0      = float(seg[time_col].iloc[idx_min])
    dur     = ROLLING_WINDOW * sample_int
    logs.append(f"フォールバック: 最小std 使用 → Time {t0:.1f}s, 継続{dur:.0f}s, 中央値={med:.6f}")
    return med, logs, t0, dur

def extract_medians_and_logs(df, time_col, ar_col, intervals, return_times=False):
    medians, logs_all, times, durs = [], [], [], []
    for i, (s,e) in enumerate(intervals,1):
        med, logs, t0, dur = extract_stable_median(df, time_col, ar_col, s, e)
        medians.append(med)
        logs_all.append((f"区間{i} ({s}s–{e}s)", logs))
        times.append(t0)
        durs.append(dur)
    if return_times:
        return medians, logs_all, times, durs
    return medians, logs_all

def calculate_precision(df_out, time_col, pco2_col, times, durs):
    """
    各区間の pCO2 標準偏差（繰り返し精度）を計算
    """
    precisions = []
    for t0, dur in zip(times, durs):
        if t0 is None or dur is None:
            precisions.append(None)
            continue
        df_seg = df_out[(df_out[time_col] >= t0) & (df_out[time_col] <= t0 + dur)]
        precisions.append(df_seg[pco2_col].std() if not df_seg.empty else None)
    return precisions
