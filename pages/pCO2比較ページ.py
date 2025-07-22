import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pingouin as pg
from analyze_stable_ar import extract_medians_and_logs, calculate_precision

# (1) 目安テーブルをあらかじめ定義
RANK_CRITERIA = {
    'S': '0–5 %（ほぼばらつきなし）',
    'A': '5–10 %（十分高い再現性）',
    'B': '10–20 %（中程度のばらつき）',
    'C': '20–30 %（注意が必要）',
    'D': '30–50 %（かなりばらつき大）',
    'E': '50 %以上（再現性低）'
}


def get_rank(cv):
    if   cv < 5:    return 'S'
    elif cv < 10:   return 'A'
    elif cv < 20:   return 'B'
    elif cv < 30:   return 'C'
    elif cv < 50:   return 'D'
    else:           return 'E'

st.set_page_config(page_title="pCO₂ 安定区間比較ページ", layout="wide")
st.title("pCO₂ 安定区間比較ページ")

# -----------------------------------------------------------------------------
# 0. 標準ガス設定 (ppm → µatm)
st.sidebar.subheader("標準ガス設定 (ppm → µatm)")
VAPOR_PRESSURE = 0.0313
PRESSURE_ATM   = 1.0
ppm1 = st.sidebar.number_input("ガス1 [ppm]", value=191)
ppm2 = st.sidebar.number_input("ガス2 [ppm]", value=568)
ppm3 = st.sidebar.number_input("ガス3 [ppm]", value=762)
ppm4 = st.sidebar.number_input("ガス4 [ppm]", value=1720)
# µatm に変換
uatm_vals = [
    (PRESSURE_ATM - VAPOR_PRESSURE) * ppm1,
    (PRESSURE_ATM - VAPOR_PRESSURE) * ppm2,
    (PRESSURE_ATM - VAPOR_PRESSURE) * ppm3,
    (PRESSURE_ATM - VAPOR_PRESSURE) * ppm4,
]

# -----------------------------------------------------------------------------
# 1. 回帰係数入力（サイドバー）
st.sidebar.subheader("回帰係数 a, b, c を入力")
a = st.sidebar.number_input("a", value=0.23490, step=1e-5, format="%.5f")
b = st.sidebar.number_input("b", value=-0.99530, step=1e-5, format="%.5f")
c = st.sidebar.number_input("c", value=0.78380, step=1e-5, format="%.5f")

# -----------------------------------------------------------------------------
# 2. 複数ファイルアップロード
st.header("1. 比較する CSV ファイルを複数選択")
uploaded_files = st.file_uploader(
    "CSV ファイルを複数選択 (Ctrl/Cmd + クリック)",
    type="csv", accept_multiple_files=True
)
if not uploaded_files:
    st.info("まずは比較したい CSV ファイルをアップロードしてください。")
    st.stop()

# 共通定数・区間定義
E1, E2, E3 = 0.00387, 2.858, 0.0181
intervals = [(0,800),(800,1600),(1600,2400),(2400,3200)]

# -----------------------------------------------------------------------------
# 3. 各ファイル読み込み＋pCO₂ 計算
all_dfs = {}
for f in uploaded_files:
    df = pd.read_csv(f)
    df['R_CO2'] = -np.log10((df['A_R'] - E1) / (E2 - E3 * df['A_R']))
    def est_pCO2(Rv):
        A,B,C = a, b, (c - Rv)
        disc = B*B - 4*A*C
        x1 = (-B + np.sqrt(disc)) / (2*A)
        x2 = (-B - np.sqrt(disc)) / (2*A)
        return 10 ** max(x1, x2)
    df['pCO2'] = df['R_CO2'].apply(est_pCO2)
    # ファイル名から拡張子 .csv を除去
    label = f.name.rsplit('.', 1)[0]
    all_dfs[label] = df

# -----------------------------------------------------------------------------
# 4. 各ファイル個別プロット
st.header("2. 各ファイルの pCO₂ 時系列プロット (Time ≥ 0s)")
for name, df in all_dfs.items():
    st.subheader(f"● {name}")
    fig, ax = plt.subplots(figsize=(6, 3))
    sub = df[df['Time'] >= 0]
    ax.plot(sub['Time'], sub['pCO2'], marker='o', markersize=3, linewidth=1, color='red', label=name)
    ax.set_title(name, fontsize=12)
    ax.set_xlabel("Time [s]", fontsize=10)
    ax.set_ylabel("pCO₂ [µatm]", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    st.pyplot(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 5. 全ファイル重ね書きプロット
st.header("3. 全ファイル重ね書きプロット (Time ≥ 0s)")
fig, ax = plt.subplots(figsize=(6, 3))
for name, df in all_dfs.items():
    sub = df[df['Time'] >= 0]
    ax.plot(sub['Time'], sub['pCO2'], marker='o', markersize=3, linewidth=1, label=name)
ax.set_title("重ね書き pCO₂", fontsize=12)
ax.set_xlabel("Time [s]", fontsize=10)
ax.set_ylabel("pCO₂ [µatm]", fontsize=10)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
st.pyplot(fig, use_container_width=True)


# 6. 繰り返し精度 (σ)
st.header("4. 繰り返し精度(σ)")
for name, df in all_dfs.items():
    # 区間ごとの medians, logs_all, times, durations を取得
    medians, logs_all, times, durations = extract_medians_and_logs(
        df, 'Time', 'A_R', intervals, return_times=True
    )
    with st.expander(f"{name} の繰り返し精度 (σ)", expanded=False):
        # pCO2 繰り返し精度
        p_sigmas = calculate_precision(df, 'Time', 'pCO2', times, durations)
        # A_R 繰り返し精度
        ar_sigmas = []
        for t0, dur in zip(times, durations):
            if t0 is None or dur is None:
                ar_sigmas.append(np.nan)
            else:
                seg = df[(df['Time'] >= t0) & (df['Time'] < t0 + dur)]
                ar_sigmas.append(seg['A_R'].std() if not seg.empty else np.nan)

        # テーブル作成
        sigma_df = pd.DataFrame({
            '区間': [label for (label, _) in logs_all],
            'σ_AR': np.round(ar_sigmas, 5),
            'σ_pCO₂': np.round(p_sigmas, 5)
        })
        st.table(sigma_df)

# 7. 分解能解析
st.header("5. 分解能解析")
for name, df in all_dfs.items():
    with st.expander(f"{name} の分解能解析", False):
        # ────────────────────────
        # ① 区間情報の再取得
        medians, logs_all, times, durations = extract_medians_and_logs(
            df, 'Time', 'A_R', intervals, return_times=True
        )
        # ② A_R の繰り返し精度 σ を計算（分解能計算で使う）
        ar_sigmas = []
        for t0, dur in zip(times, durations):
            if t0 is None or dur is None:
                ar_sigmas.append(np.nan)
            else:
                seg = df[(df['Time'] >= t0) & (df['Time'] < t0 + dur)]
                ar_sigmas.append(seg['A_R'].std() if not seg.empty else np.nan)

        # ③ 分解能計算
        res = []
        for i in range(len(times)):
            # 区間0 または中央値が None、あるいは σ が NaN のときは NaN
            if i == 0 or medians[i] is None or medians[i-1] is None or np.isnan(ar_sigmas[i]):
                res.append(np.nan)
            else:
                # ここでサイドバーで定義した uatm_vals を使う
                dp    = uatm_vals[i] - uatm_vals[i-1]
                dAR   = medians[i]   - medians[i-1]
                digit = 10 ** np.floor(np.log10(ar_sigmas[i])) if ar_sigmas[i] > 0 else 0
                raw   = abs(dp / dAR) * digit
                # 有効数字1桁で丸め
                sigd  = -int(np.floor(np.log10(raw))) if raw > 0 else 0
                res.append(round(raw, sigd))

        # ④ 表示用 DataFrame を組み立て
        df_res = pd.DataFrame({
            '区間': [label for (label, _) in logs_all],
            '分解能 [µatm]': res
        })
        st.table(df_res)

# 8. 応答速度 t₉₀
st.header("6. 応答速度 (t₉₀)")
for name, df in all_dfs.items():
    with st.expander(f"{name} の応答速度", False):
        for idx,(s,e) in enumerate(intervals,1):
            seg = df[(df['Time']>=s)&(df['Time']<=e)]
            if seg.empty:
                st.write(f"- 区間{idx}: データなし")
                continue
            p0 = seg['pCO2'].iloc[0]
            ps = seg['pCO2'].tail(10).median()
            d = abs(p0-ps)
            thr = p0 - 0.9*d if ps<p0 else p0 + 0.9*d
            hit = seg[(seg['pCO2']<=thr) if ps<p0 else (seg['pCO2']>=thr)]
            if not hit.empty:
                t90 = hit['Time'].iloc[0]-s
                st.write(f"- 区間{idx}: t₉₀ = {t90:.1f} s")
            else:
                st.write(f"- 区間{idx}: 未到達")

# -----------------------------------------------------------------------------
# 6. 縦長データへの集約 & 再現性評価
records = []
for name, df in all_dfs.items():
    medians, logs_all, times, durations = extract_medians_and_logs(
        df, 'Time', 'A_R', intervals, return_times=True
    )
    # 各区間 t90, σ, 分解能
    t90s      = []
    sigmas    = calculate_precision(df, 'Time', 'pCO2', times, durations)
    raw_res   = []
    for i,(s,e) in enumerate(intervals):
        # t90
        seg = df[(df['Time']>=s)&(df['Time']<=e)]
        if seg.empty or medians[i] is None:
            t90s.append(np.nan)
        else:
            p0 = seg['pCO2'].iloc[0]
            ps = seg['pCO2'].tail(10).median()
            d  = abs(p0-ps)
            thr = (p0-0.9*d) if ps<p0 else (p0+0.9*d)
            hit = seg[seg['pCO2']<=thr] if ps<p0 else seg[seg['pCO2']>=thr]
            t90s.append((hit['Time'].iloc[0]-s) if not hit.empty else np.nan)
        # 分解能
        if i==0 or medians[i] is None or medians[i-1] is None or np.isnan(sigmas[i]):
            raw_res.append(np.nan)
        else:
            dp  = uatm_vals[i] - uatm_vals[i-1]
            dAR = medians[i] - medians[i-1]
            digit = 10**np.floor(np.log10(sigmas[i])) if sigmas[i]>0 else 0
            r = abs(dp/dAR)*digit
            sd = -int(np.floor(np.log10(r))) if r>0 else 0
            raw_res.append(round(r, sd))
    for idx,(label,med) in enumerate(zip([f"{i+1} ({s}–{e}s)" for i,(s,e) in enumerate(intervals)], medians)):
        records.append({
            'file': name,
            '区間': label,
            'pCO₂_median': med,
            't90 [s]': t90s[idx],
            'σ_pCO₂ [µatm]': sigmas[idx],
            '分解能 [µatm]': raw_res[idx]
        })



# 9. 再現性評価：変動係数 (CV %

# ──────────────────────────────────────────────────────
# (2) 画面の先頭かサイドバーで「ランク目安」表を表示
st.sidebar.subheader("ランク評価の目安")
criteria_df = pd.DataFrame([
    {'ランク': r, 'CV範囲': rng}
    for r, rng in RANK_CRITERIA.items()
])
st.sidebar.table(criteria_df)

# ──────────────────────────────────────────────────────
# （既存処理で CV を計算後）
# df_rep には既に '区間', 'pCO₂_median', 't90 [s]', 'σ_pCO₂ [µatm]' が入っている想定
df_rep = pd.DataFrame(records)
# 各 CV 列に対してランク列を追加
df_rep['Rank_med']  = df_rep['pCO₂_median'].pct_change().abs().apply(lambda x: get_rank(x*100))
df_rep['Rank_t90']  = df_rep['t90 [s]'].apply(lambda x: get_rank(x/df_rep['t90 [s]'].mean()*100))
df_rep['Rank_sig']  = df_rep['σ_pCO₂ [µatm]'].apply(lambda x: get_rank(x/df_rep['σ_pCO₂ [µatm]'].mean()*100))

# ここはお好みで「CV_med」テーブルを作ってもよいです
cv = df_rep.groupby('区間').agg(
    CV_med=('pCO₂_median', lambda x: x.std()/x.mean()*100),
    CV_t90=('t90 [s]',       lambda x: x.std()/x.mean()*100),
    CV_sig=('σ_pCO₂ [µatm]', lambda x: x.std()/x.mean()*100)
).reset_index()

# ランク列を追加
# 2) ランク列を追加
cv['Rank_med'] = cv['CV_med'].apply(get_rank)
cv['Rank_t90'] = cv['CV_t90'].apply(get_rank)
cv['Rank_sig'] = cv['CV_sig'].apply(get_rank)

# 3) パーセント表示用に文字列化（ランク列には影響しない）
for col in ['CV_med','CV_t90','CV_sig']:
    cv[col] = cv[col].map(lambda x: f"{x:.1f}%")

st.header("7. 再現性評価：変動係数 (CV %)")
st.markdown(
    "この表は，同じ区間を複数回測定したときに、\n"
    "- pCO₂ 中央値\n"
    "- 応答速度 t₉₀\n"
    "- pCO₂ の繰り返し精度σ \n"
    "がどれくらいばらつくか（変動係数 CV%）を示しています。"
)

# CV とランクをまとめて表示
st.table(cv)
st.markdown("**各ランク評価**: S > A > B > C > D > E（E が最も再現性低）")

# 10. ICC（pCO₂ 中央値）
icc = pg.intraclass_corr(
    data=df_rep.dropna(subset=['pCO₂_median']),
    targets='区間', raters='file', ratings='pCO₂_median'
)

st.sidebar.subheader("🗂 比較対象ファイル一覧")
for name in all_dfs.keys():
    st.sidebar.write(f"- {name}")

icc_records = []
for name, df in all_dfs.items():
    # extract_medians_and_logs で定常開始 times, durations を取得済み
    medians, logs_all, times, durations = extract_medians_and_logs(
        df, 'Time', 'A_R', intervals, return_times=True
    )
    for i, ((start, end), t0) in enumerate(zip(intervals, times), start=1):
        if t0 is None:
            continue
        # 定常開始時刻 t0 ～ 区間終了 end までの pCO2
        seg = df[(df['Time'] >= t0) & (df['Time'] <= end)]
        for _, row in seg.iterrows():
            icc_records.append({
                'interval': f"{i} ({start}–{end}s)",
                'file': name,
                'pCO2': row['pCO2']
            })

icc_df = pd.DataFrame(icc_records)

# ―――― 9. ICC 計算 ――――
icc = pg.intraclass_corr(
    data=icc_df,
    targets='interval',    # 区間ごとに一致度を測る
    raters='file',         # ファイルを評価者に見立てる
    ratings='pCO2'         # 生データをそのまま評価値に
)

st.header("8. ICC — pCO₂ 中央値")
st.markdown("""
この表では，区間内で得られた pCO₂ 中央値の一致度を  クラス内相関係数（ICC）で評価しています。  
ICC が 1 に近いほど，ファイル間で非常によく一致していることを示します。  
""")
st.dataframe(
    icc[['Type','ICC','CI95%','F','pval']].round(3),
    use_container_width=True
)