import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("流路長ごとの比較")

# --- 標準ガス設定 (ppm → µatm) はサイドバーに配置
st.sidebar.subheader("標準ガス設定 (ppm → µatm)")
VAPOR_PRESSURE = 0.0313
PRESSURE_ATM   = 1.0
ppm1 = st.sidebar.number_input("ガス1 [ppm]", value=191)
ppm2 = st.sidebar.number_input("ガス2 [ppm]", value=568)
ppm3 = st.sidebar.number_input("ガス3 [ppm]", value=762)
ppm4 = st.sidebar.number_input("ガス4 [ppm]", value=1720)
uatm_values = [(PRESSURE_ATM - VAPOR_PRESSURE) * ppm for ppm in [ppm1, ppm2, ppm3, ppm4]]

# --- 回帰係数入力（ページ内に表示）
st.subheader("流路長ごとの回帰係数 a, b, c 入力")
coeff_dict = {}
for L in [200, 400, 600, 800]:
    st.markdown(f"**{L} mm**")
    c1, c2, c3 = st.columns(3)
    a_val = c1.number_input(f"a_{L}", value=0.23490, step=1e-5, format="%.5f", key=f"a_{L}")
    b_val = c2.number_input(f"b_{L}", value=-0.99530, step=1e-5, format="%.5f", key=f"b_{L}")
    c_val = c3.number_input(f"c_{L}", value=0.78380, step=1e-5, format="%.5f", key=f"c_{L}")
    coeff_dict[L] = (a_val, b_val, c_val)

# --- CSVファイルアップロード
st.header("1. 比較する CSV ファイルを複数選択")
uploaded_files = st.file_uploader(
    "CSV ファイルを複数選択 (Ctrl/Cmd + クリック)",
    type="csv", accept_multiple_files=True
)
if not uploaded_files:
    st.info("まずは比較したい CSV ファイルをアップロードしてください。")
    st.stop()

# --- 共通定数（Rv計算用）
E1, E2, E3 = 0.00387, 2.858, 0.0181

# --- pCO₂変換関数
def est_pCO2(Rv, a, b, c):
    A, B, C = a, b, (c - Rv)
    disc = B * B - 4 * A * C
    x1 = (-B + np.sqrt(disc)) / (2 * A)
    return 10 ** x1

# --- ファイル処理
all_dfs = {}
for f in uploaded_files:
    df = pd.read_csv(f)
    df['R_CO2'] = -np.log10((df['A_R'] - E1) / (E2 - E3 * df['A_R']))

    # ファイル名から流路長を判定
    length_detected = None
    for L in coeff_dict.keys():
        if f"{L}" in f.name:  # ファイル名に "200", "400" などが含まれるか
            length_detected = L
            break

    if length_detected is None:
        st.warning(f"{f.name} の流路長が認識できません（デフォルト600mm係数を使用）")
        length_detected = 600

    a, b, c = coeff_dict[length_detected]
    df['pCO2'] = df['R_CO2'].apply(lambda Rv: est_pCO2(Rv, a, b, c))

    label = f"{f.name.rsplit('.', 1)[0]} ({length_detected}mm)"
    all_dfs[label] = df

# --- 各ファイル個別プロット
st.header("2. 各ファイルの pCO₂ 時系列プロット")
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


# --- 全ファイル重ね書き
st.header("3. 全ファイル重ね書きプロット")

# --- ファイルから含まれる流路長を検出
lengths_detected = set()
for name in all_dfs.keys():
    for L in coeff_dict.keys():
        if f"{L}" in name:
            lengths_detected.add(L)

# --- 流路長ごとに日付選択
st.subheader("流路長ごとの日付設定")
length_date_map = {}
for L in sorted(lengths_detected):
    date_selected = st.date_input(f"{L} mm の測定日付", key=f"date_{L}")
    length_date_map[L] = date_selected.strftime("%Y%m%d") if date_selected else "未選択"

# --- プロット
fig, ax = plt.subplots(figsize=(6, 3))
for name, df in all_dfs.items():
    sub = df[df['Time'] >= 0].sort_values(by='Time')

    # ファイル名から流路長抽出
    length_detected = None
    for L in coeff_dict.keys():
        if f"{L}" in name:
            length_detected = L
            break
    if length_detected is None:
        length_detected = "unknown"

    # 日付を凡例に反映
    date_suffix = length_date_map.get(length_detected, "未選択")
    legend_label = f"{length_detected}mm-{date_suffix}"

    ax.plot(sub['Time'], sub['pCO2'], marker='o', markersize=3, linewidth=1, label=legend_label)

# --- タイトルは固定
ax.set_title("pCO₂ Calculation Results", fontsize=12)
ax.set_xlabel("Time [s]", fontsize=10)
ax.set_ylabel("pCO₂ [µatm]", fontsize=10)
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)
st.pyplot(fig, use_container_width=True)