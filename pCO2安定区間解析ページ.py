import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import io
import os
import zipfile
import streamlit as st
from streamlit.components.v1 import html
from analyze_stable_ar import extract_medians_and_logs, calculate_precision

intervals = [(0,800),(800,1600),(1600,2400),(2400,3200)]

# Henryの法則定数
VAPOR_PRESSURE = 0.0313  # atm
PRESSURE_ATM   = 1.0     # atm

def convert_ppm(ppm):
    return (PRESSURE_ATM - VAPOR_PRESSURE) * ppm

# セッション初期化
if 'results' not in st.session_state:
    st.session_state.results = None

st.title("pCO₂安定区間解析アプリ")

# ① 標準ガス ppm → µatm 変換
st.subheader("① 標準ガス (ppm → µatm) 変換")
st.write("25℃・1 atm における換算式：pCO₂ [µatm] = (1 - 0.0313) × ppm")
cols = st.columns(4)
defaults = [191, 568, 762, 1720]
ppm_values, uatm_values = [], []
for i, col in enumerate(cols):
    ppm = col.number_input(f"標準ガス{i+1} (ppm)", value=defaults[i], step=1, key=f"ppm_{i}")
    uatm = convert_ppm(ppm)
    col.markdown(f"→ **{uatm:.1f} µatm**")
    ppm_values.append(ppm)
    uatm_values.append(uatm)

# --- 複数ファイルアップロード ---
st.subheader("② 測定CSVファイルのアップロード")
uploaded_files = st.file_uploader(
    "CSVファイルを選択（複数可）",
    type="csv",
    accept_multiple_files=True
)
if not uploaded_files:
    st.stop()

# --- ファイル名だけのリストを作成してセレクトボックスで切り替え ---
filenames = [f.name for f in uploaded_files]
sel_name  = st.selectbox("解析するファイルを選択", filenames)
# 選ばれた名前と同じ UploadedFile オブジェクトを取得
sel_file  = next(f for f in uploaded_files if f.name == sel_name)

# --- ここからは “sel_file” のみを処理するので、タブよりもクリーン ---
base = os.path.splitext(sel_file.name)[0]
df   = pd.read_csv(sel_file)

st.markdown(f"### ファイル：{sel_file.name}")
st.dataframe(df.head(5))

# デフォルト列インデックス
cols = list(df.columns)
idx_time = cols.index("Time") if "Time" in cols else 0
idx_ar   = cols.index("A_R")   if "A_R"   in cols else 0

time_col = st.selectbox("Time 列", cols, index=idx_time)
ar_col   = st.selectbox("A_R 列",   cols, index=idx_ar)


# ③ 安定区間解析
st.subheader("③ A_R 安定区間抽出（固定4区間）")
if st.button("安定区間を解析", key=f"analyze_{base}"):
    intervals = [(0,800),(800,1600),(1600,2400),(2400,3200)]
    medians, logs_all, stable_times, durations = extract_medians_and_logs(
        df, time_col, ar_col, intervals, return_times=True
    )
    st.session_state.results = {
        'df': df,
        'time_col': time_col,
        'ar_col': ar_col,
        'medians': medians,
        'logs_all': logs_all,
        'stable_times': stable_times,
        'durations': durations,
        'ppm_values': ppm_values,
        'uatm_values': uatm_values,
        'intervals': intervals
    }

# セッションから結果を取得
res = st.session_state.results
if res:
    df           = res['df']
    time_col     = res['time_col']
    ar_col       = res['ar_col']
    medians      = res['medians']
    logs_all     = res['logs_all']
    stable_times = res['stable_times']
    durations    = res['durations']
    ppm_values   = res['ppm_values']
    uatm_values  = res['uatm_values']

    # 解析ログ & A_R中央値
    st.subheader("解析ログ & A_R 中央値")
    for i, ((label, logs), med) in enumerate(zip(logs_all, medians), 1):
        st.markdown(f"### {label}")
        for line in logs:
            st.text(line)
        if med is not None:
            st.markdown(f"**区間{i} 中央値：{med:.6f}**")
        else:
            st.warning(f"区間{i}: 有効な中央値が見つかりませんでした")

    

    # 安定区間サマリー
    st.subheader("安定区間サマリー")
    adf_list, pval_list, status_list = [], [], []
    for _, logs in logs_all:
        time_logs = [l for l in logs if l.strip().startswith("Time")]
        if time_logs:
            line = time_logs[-1]
            parts = line.split("|")[1].strip().split(", ")
            adf    = float(parts[0].split("=")[1])
            pval   = float(parts[1].split("=")[1])
            status = parts[2]
        else:
            adf, pval, status = None, None, None
        adf_list.append(round(adf,3)    if adf   is not None else None)
        pval_list.append(round(pval,3)  if pval  is not None else None)
        status_list.append(status)

    # ② DataFrame 作成。uatm_values はセッションから取得済みと仮定
    labels = [f"区間{i}" for i in range(1, len(medians)+1)]
    stable_summary_df = pd.DataFrame({
        "区間":            labels,
        "開始時間 [s]":     stable_times,
        "継続時間 [s]":     durations,
        "A_R 中央値":      [round(m,6) if m is not None else None for m in medians],
        "pCO₂ [µatm]":     [round(v,3) if v is not None else None for v in uatm_values],
        "ADF値":           adf_list,
        "p値":             pval_list,
        "定常判定":         status_list,
    })

    # ③ 表示＆ダウンロード
    st.table(stable_summary_df)
    original_name = sel_file.name
    base = os.path.splitext(os.path.basename(original_name))[0]
    download_name = f"定常 {base}.csv"
    csv_data = stable_summary_df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
    st.download_button("CSV でダウンロード",     data=csv_data,   file_name=f"定常 {base} .csv", mime="text/csv")




    # ④-1 A_R 時系列と安定区間
    st.subheader("④-1 A_R 時系列と安定区間")
    fig1, ax1 = plt.subplots(figsize=(8,3))
    ax1.plot(df[time_col], df[ar_col], label="A_R")
    for t in stable_times:
        if t is not None:
            ax1.axvline(x=t, linestyle="--", linewidth=1)
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("A_R")
    st.pyplot(fig1)


    # ④-2 A_R 縦線なし
    st.subheader("④-2 A_R 時系列プロット（縦線なし）")
    fig1b, ax1b = plt.subplots(figsize=(8,3))
    ax1b.plot(df[time_col], df[ar_col], color="blue", label="A_R")
    ax1b.set_xlabel("Time [s]")
    ax1b.set_ylabel("A_R")
    ax1b.legend()
    st.pyplot(fig1b)

    # ⑤ 回帰係数 a, b, c 推定
    st.subheader("⑤ 回帰係数 a, b, c 推定")
    med_vals  = medians[1:4]
    uatm_vals = uatm_values[1:4]
    coeffs = None
    if all(m is not None for m in med_vals):
        e1, e2, e3 = 0.00387, 2.858, 0.0181
        arr  = np.array(med_vals)
        Rv   = -np.log10((arr - e1) / (e2 - e3 * arr))
        logp = np.log10(np.array(uatm_vals))
        X = np.vstack([logp**2, logp, np.ones(len(logp))]).T
        a, b, c = np.linalg.lstsq(X, Rv, rcond=None)[0]
        st.markdown(f"**a={a:.4f}, b={b:.4f}, c={c:.4f}**")
        coeffs = (a, b, c)
    else:
        st.warning("abc推定に必要な中央値が不足しています")
        

    # ⑥ 全データの pCO₂ 計算 & CSV出力
    if coeffs:
        a, b, c = coeffs
        e1, e2, e3 = 0.00387, 2.858, 0.0181

        def calc_R(x):
            return -np.log10((x - e1) / (e2 - e3 * x))

        def est_pCO2(Rv):
            A, B, C = a, b, c - Rv
            disc = B*B - 4*A*C
            if disc < 0:
                return np.nan
            # 両解を計算
            sqrt_disc = np.sqrt(disc)
            r1 = (-B + sqrt_disc) / (2*A)
            #r2 = (-B - sqrt_disc) / (2*A)
            #root = max(r1, r2)  # 物理的に妥当な（大きい方の）解を選択
            return 10**r1

        df_out = df.copy()
        df_out["R_CO2"]          = df_out[ar_col].apply(calc_R)
        df_out["pCO2_estimated"] = df_out["R_CO2"].apply(est_pCO2)

        # CSV出力
        csv = df_out.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="pCO₂ 推定結果を CSV でダウンロード",
            data=csv,
            file_name=f"pCO2 {base} .csv",
            mime="text/csv"
    )

        # ⑦ pCO₂ プロット (カスタム範囲)
        st.subheader("⑦ pCO₂ プロット (カスタム範囲)")
        default_xmin = 0
        default_xmax = int(df_out[time_col].max())
        x_min2 = st.number_input("pCO₂: x軸最小 (Time [s])",
                                 value=default_xmin, min_value=0, key="pco2_xmin")
        x_max2 = st.number_input("pCO₂: x軸最大 (Time [s])",
                                 value=default_xmax, max_value=default_xmax, key="pco2_xmax")

        mask = df_out[time_col] >= 0
        default_ymin = float(df_out.loc[mask, "pCO2_estimated"].min())
        default_ymax = float(df_out.loc[mask, "pCO2_estimated"].max())

        y_min2 = st.number_input("pCO₂: y軸最小 (µatm)",
                                 value=default_ymin, key="pco2_ymin")
        y_max2 = st.number_input("pCO₂: y軸最大 (µatm)",
                                 value=default_ymax, key="pco2_ymax")

        fig2b, ax2b = plt.subplots(figsize=(8,3))
        ax2b.plot(df_out[time_col], df_out["pCO2_estimated"], color="red", label="pCO₂")
        ax2b.set_xlim(x_min2, x_max2)
        ax2b.set_ylim(y_min2, y_max2)
        ax2b.set_xlabel("Time [s]")
        ax2b.set_ylabel("pCO₂ [µatm]")
        ax2b.legend()
        st.pyplot(fig2b)

        # ⑧ 吸光度比 (A_R) プロット (カスタム範囲)
        st.subheader("⑧ 吸光度比 (A_R) プロット (カスタム範囲)")
        ar_default_xmin = 0
        ar_default_xmax = int(df[time_col].max())
        ar_xmin = st.number_input("A_R: x軸最小 (Time [s])",
                                  value=ar_default_xmin, min_value=0, key="ar_xmin")
        ar_xmax = st.number_input("A_R: x軸最大 (Time [s])",
                                  value=ar_default_xmax, max_value=ar_default_xmax, key="ar_xmax")

        mask_ar = df[time_col] >= 0
        ar_default_ymin = float(df.loc[mask_ar, ar_col].min())
        ar_default_ymax = float(df.loc[mask_ar, ar_col].max())

        ar_ymin = st.number_input("A_R: y軸最小",
                                  value=ar_default_ymin, key="ar_ymin")
        ar_ymax = st.number_input("A_R: y軸最大",
                                  value=ar_default_ymax, key="ar_ymax")

        fig_ar, ax_ar = plt.subplots(figsize=(8,3))
        ax_ar.plot(df[time_col], df[ar_col], color="blue", label="A_R")
        ax_ar.set_xlim(ar_xmin, ar_xmax)
        ax_ar.set_ylim(ar_ymin, ar_ymax)
        ax_ar.set_xlabel("Time [s]")
        ax_ar.set_ylabel("A_R")
        ax_ar.legend()
        st.pyplot(fig_ar)

        # ⑨ 繰り返し精度 a.k.a 測定精度 (A_R & pCO₂)
        st.subheader("⑨ 繰り返し精度 a.k.a 測定精度 (A_R & pCO₂)")
        ar_precisions = []
        for t0, dur in zip(stable_times, durations):
            if t0 is None or dur is None:
                ar_precisions.append(None)
            else:
                seg_ar = df[(df[time_col] >= t0) & (df[time_col] <= t0 + dur)]
                ar_precisions.append(seg_ar[ar_col].std() if not seg_ar.empty else None)
        pco2_precisions = calculate_precision(
            df_out, time_col, "pCO2_estimated", stable_times, durations)

        for i, (sigma_ar, sigma_pco2) in enumerate(
                zip(ar_precisions, pco2_precisions), 1):
            col1, col2 = st.columns(2)
            with col1:
                if sigma_ar is not None:
                    st.metric(label=f"区間{i} σ_AR", value=f"{sigma_ar:.4f}")
                else:
                    st.warning(f"区間{i}: σ_AR 計算できませんでした")
            with col2:
                if sigma_pco2 is not None:
                    st.metric(label=f"区間{i} σ_pCO₂", value=f"{sigma_pco2:.2f} µatm")
                else:
                    st.warning(f"区間{i}: σ_pCO₂ 計算できませんでした")

        # ⑩ 分解能解析
        st.subheader("⑩ 分解能解析")
        sigma_ar = ar_precisions
        pco2_meds = uatm_values
        resolutions = [None]
        # 各区間の計算
        for i in range(1, len(medians)):
            r_prec = sigma_ar[i]
            if medians[i] is not None and medians[i-1] is not None and r_prec is not None:
                dp = pco2_meds[i] - pco2_meds[i-1]
                dAR = medians[i] - medians[i-1]
                digit = 10 ** math.floor(math.log10(r_prec)) if r_prec > 0 else 0
                raw_res = abs(dp / dAR) * digit
                sig_digit = -int(math.floor(math.log10(abs(raw_res)))) if raw_res != 0 else 0
                res_val = round(raw_res, sig_digit)
                resolutions.append(res_val)
            else:
                resolutions.append(None)
        # 計算式表示
        st.subheader("分解能計算式 (数値代入例)")
        for i in range(1, len(medians)):
            if resolutions[i] is not None:
                dp = pco2_meds[i] - pco2_meds[i-1]
                dAR = medians[i] - medians[i-1]
                digit = 10 ** math.floor(math.log10(sigma_ar[i])) if sigma_ar[i] > 0 else 0
                st.markdown(
                    f"- 区間{i}: ({dp:.3f} µatm / {dAR:.3f}) × {digit:.0e} = **{resolutions[i]:.3f} µatm**"
                )
        # 表示用丸め
        res_rounded = [x if x is None else round(x,4) for x in resolutions]
        res_df = pd.DataFrame({
            "区間": [str(i+1) for i in range(len(medians))],
            "分解能 [µatm]": res_rounded
        })
        st.dataframe(res_df)


        # ⑪ 応答時間 (Response Time, t₉₀) の完全自動化
        st.subheader("⑪ 応答時間 (Response Time, t₉₀)")
        step_times = [intervals[i][0] for i in (1,2,3)]
        response_times = []
        formulas = []

        for idx, step_time in enumerate(step_times, start=1):
            st_time = stable_times[idx]
            if st_time is None:
                response_times.append(None)
                formulas.append(None)
                continue

            post = df_out[df_out[time_col] > step_time]
            if post.empty:
                response_times.append(None)
                formulas.append(None)
                continue

            initial    = post["pCO2_estimated"].iloc[0]
            nearest_idx = (df_out[time_col] - st_time).abs().idxmin()
            stable_val  = df_out.at[nearest_idx, "pCO2_estimated"]
            threshold   = initial + 0.9 * (stable_val - initial)

            if stable_val >= initial:
                cross = post[post["pCO2_estimated"] >= threshold]
            else:
                cross = post[post["pCO2_estimated"] <= threshold]

            if cross.empty:
                response_times.append(None)
                formulas.append(None)
                continue

            t90 = cross[time_col].iloc[0] - step_time
            response_times.append(t90)

            # 数値代入済み式を作成
            formula = (
                f"{initial:.1f} + 0.9×({stable_val:.1f}−{initial:.1f}) = {threshold:.1f}"
                f" → t₉₀ = {t90:.1f}s"
            )
            formulas.append(formula)

            st.metric(label=f"{step_time}s の t₉₀", value=f"{t90:.1f} 秒")
            st.markdown(f"**式:** {formula}")

        # 区間1用にパディング
        padded_t90     = [None] + response_times
        padded_formula = [None] + formulas

        # ⑫ 全体結果まとめ
        st.subheader("⑫ 全体結果まとめ")
        labels   = [f"区間{i+1}" for i in range(len(medians))]
        ar_meds  = [round(m, 6) if m is not None else None for m in medians]
        p_refs   = [round(u, 1) for u in uatm_values]
        sig_ar   = [round(x, 4) if x is not None else None for x in ar_precisions]
        sigma_pco2    = [round(x, 2) if x is not None else None for x in pco2_precisions]
        res_vals = [round(x, 3) if x is not None else None for x in resolutions]
        t90_vals = [round(x, 1) if x is not None else None for x in padded_t90]
        formulas = padded_formula

        adf_stats, p_values, statuses = [], [], []
        for _, logs in logs_all:
            if logs:
                line = next((l for l in logs if l.startswith("Time")), None)
                if line:
                    parts = line.split('|')
                    stat_str = parts[1].strip().split(', ')
                    adf = float(stat_str[0].split('=')[1])
                    pval = float(stat_str[1].split('=')[1])
                    status = stat_str[2]
                else:
                    adf, pval, status = None, None, None
            else:
                adf, pval, status = None, None, None
            adf_stats.append(round(adf,3) if adf is not None else None)
            p_values.append(round(pval,3) if pval is not None else None)
            statuses.append(status)
        # DataFrame作成
        labels   = [f"区間{i+1}" for i in range(len(medians))]
        summary_df = pd.DataFrame({
            "区間":            labels,
            "σ_AR":           sig_ar,
            "σ_pCO₂":         sigma_pco2,
            "分解能":         res_vals,   # 必要なら計算ロジックを追加
            "t₉₀[秒]":       t90_vals,    # 区間数分の None リスト  # 必要なら計算ロジックを追加
        })
        # 全体統計シート
        global_df = pd.DataFrame({
            "項目": ["A_R 最大", "A_R 最小", "pCO₂ 最大", "pCO₂ 最小", "回帰係数 a b c"],
            "値":  [ar_ymax, ar_ymin, y_max2, y_max2, coeffs],
            "時刻[秒]": [ar_xmax, ar_xmin, x_max2, x_min2, None]
        })
        st.dataframe(summary_df)
        csv_buffer = io.StringIO()

        # ① Summary 部分
        csv_buffer.write("=== Summary ===\n")
        summary_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_buffer.write("\n")  # 空行

        # ② GlobalStats 部分
        csv_buffer.write("=== GlobalStats ===\n")
        global_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')

        # バイト列化
        csv_bytes = csv_buffer.getvalue().encode('utf-8-sig')

        # ダウンロードボタン
        st.download_button(
            label="結果まとめ＆全体統計をCSVでダウンロード",
            data=csv_bytes,
            file_name=f"summary_{base}.csv",
            mime="text/csv"
        )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_out.to_excel(writer,           sheet_name="pCO2_results",   index=False)
            summary_df.to_excel(writer,       sheet_name="Summary",        index=False)
            global_df.to_excel(writer,        sheet_name="回帰係数 a,b,c", index=False)
            stable_summary_df.to_excel(writer, sheet_name="定常区間",       index=False)
        # ← ここで with ブロックを抜けると自動保存されるので save() は不要

        excel_data = output.getvalue()

        # 2) ダウンロードボタン
        st.download_button(
            label="全結果を Excel でダウンロード",
            data=excel_data,
            file_name=f"all_sheets_{base}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"dl_{base}"
        )

