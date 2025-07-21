import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="測定条件管理", layout="wide")
st.title("測定条件管理ページ")

# --- CSV アップロード & 読み込むボタン ---
st.markdown("**純水測定データCSVをアップロード後、『読み込む』を押すと I/S/R が自動計算されます**")
col1, col2 = st.columns([3,1])
with col1:
    uploaded_data = st.file_uploader("純水測定データCSVを選択", type="csv", key="auto_calc")
with col2:
    load_btn = st.button("読み込む")

# 波長リストとプレースホルダー
wavelengths = [435, 490, 590, 735]
# セッションからプレースホルダーを取得
auto_I = st.session_state.get('auto_I', [0.0] * 4)
auto_S = st.session_state.get('auto_S', [0] * 4)
auto_R = st.session_state.get('auto_R', [0] * 4)

# --- 読み込み & 自動計算 ---
if load_btn:
    if uploaded_data is None:
        st.warning("先に CSV ファイルをアップロードしてください")
    else:
        try:
            content = uploaded_data.getvalue().decode('utf-8')
            df = pd.read_csv(io.StringIO(content))
        except Exception as e:
            st.error(f"CSV の読み込みに失敗しました: {e}")
        else:
            st.success("ファイル読み込み完了。I/S/R を計算しています…")
            new_I, new_S, new_R = [0.0]*4, [0]*4, [0]*4
            for i, wl in enumerate(wavelengths):
                col_I = f"I_{wl}nm"
                if col_I in df.columns:
                    vals = df[col_I].dropna()
                    if len(vals) >= 1:
                        new_I[i] = float(vals.tail(5).mean())
                col_S = f"Sig_{wl}nm"
                if col_S in df.columns:
                    s_vals = df[col_S].dropna()
                    if not s_vals.empty:
                        new_S[i] = int(s_vals.iloc[-1])
                col_R = f"Ref_{wl}nm"
                if col_R in df.columns:
                    r_vals = df[col_R].dropna()
                    if not r_vals.empty:
                        new_R[i] = int(r_vals.iloc[-1])
            st.session_state['auto_I'] = new_I
            st.session_state['auto_S'] = new_S
            st.session_state['auto_R'] = new_R
            auto_I, auto_S, auto_R = new_I, new_S, new_R

# --- フォーム入力 & 保存 ---
if 'measure_conditions' not in st.session_state:
    st.session_state['measure_conditions'] = {}

with st.form(key='measure_form'):
    measure_date = st.date_input("測定日")
    recorder = st.selectbox("記入者名", options=["山本空", "甲彩希"])
    room_temp = st.number_input("室温 [℃]", value=25.0)
    water_temp = st.number_input("水温 [℃]", value=25.0)

    ppm1 = st.number_input("pCO₂ [ppm] 標準ガス1", value=193)
    ppm2 = st.number_input("pCO₂ [ppm] 標準ガス2", value=373)
    ppm3 = st.number_input("pCO₂ [ppm] 標準ガス3", value=759)
    ppm4 = st.number_input("pCO₂ [ppm] 標準ガス4", value=1710)
    ph_conc = st.number_input("pH指示薬濃度 [µmol/l]", value=15.0)
    gas_flow = st.number_input("ガス流量 [L/min]", value=0.5)
    flow_ml = st.number_input("流量 [ml/min]", value=0.4)
    beaker_vol = st.number_input("ビーカー容量 [L]", value=1.0)
    beaker_sol = st.text_input("ビーカー内溶液", value="海水")

    Iv1 = st.number_input("I値1 (435 nm)", value=auto_I[0])
    Iv2 = st.number_input("I値2 (490 nm)", value=auto_I[1])
    Iv3 = st.number_input("I値3 (590 nm)", value=auto_I[2])
    Iv4 = st.number_input("I値4 (735 nm)", value=auto_I[3])

    Sv1 = st.number_input("S値1 (435 nm)", value=auto_S[0], step=1, format="%d")
    Sv2 = st.number_input("S値2 (490 nm)", value=auto_S[1], step=1, format="%d")
    Sv3 = st.number_input("S値3 (590 nm)", value=auto_S[2], step=1, format="%d")
    Sv4 = st.number_input("S値4 (735 nm)", value=auto_S[3], step=1, format="%d")

    Rv1 = st.number_input("R値1 (435 nm)", value=auto_R[0], step=1, format="%d")
    Rv2 = st.number_input("R値2 (490 nm)", value=auto_R[1], step=1, format="%d")
    Rv3 = st.number_input("R値3 (590 nm)", value=auto_R[2], step=1, format="%d")
    Rv4 = st.number_input("R値4 (735 nm)", value=auto_R[3], step=1, format="%d")

    gas_unit = st.selectbox("ガス交換ユニット材料", options=["PDMS and シリコーン膜","AF"])
    channel_len = st.selectbox("流路長 [mm]", options=[200,300,400,500,600,700,800,864], index=7)
    abc_vals = st.text_input("回帰係数 a/b/c", value="0/0/0")
    notes = st.text_area("備考")

    submit = st.form_submit_button("保存")

if submit:
    st.session_state['measure_conditions'] = {
        'date': str(measure_date),
        'recorder': recorder,
        'room_temperature': room_temp,
        'water_temperature': water_temp,
        'ppm_values': [ppm1, ppm2, ppm3, ppm4],
        'ph_concentration': ph_conc,
        'gas_flow': gas_flow,
        'flow_ml': flow_ml,
        'beaker_volume': beaker_vol,
        'beaker_solution': beaker_sol,
        'I_values': [Iv1, Iv2, Iv3, Iv4],
        'S_values': [Sv1, Sv2, Sv3, Sv4],
        'R_values': [Rv1, Rv2, Rv3, Rv4],
        'gas_unit': gas_unit,
        'channel_length': channel_len,
        'abc': abc_vals,
        'notes': notes
    }
    st.success("測定条件を保存しました！")

# --- 保存済み条件の表示 & Excelダウンロード ---
mc = st.session_state['measure_conditions']
if mc:
    # 基本情報表示省略...

    # --- Excel ダウンロード ---
    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # 基本情報を DataFrame に（横幅）
        VAPOR_PRESSURE = 0.0313
        PRESSURE_ATM = 1.0
        uatm_vals = [(PRESSURE_ATM - VAPOR_PRESSURE) * x for x in mc['ppm_values']]
        info_df = pd.DataFrame({
            '項目': ['測定日','記入者','室温 [℃]','水温 [℃]','pCO₂ [ppm]','pCO₂ [µatm]','pH指示薬濃度 [µmol/l]',
                    'ガス流量 [L/min]','流量 [ml/min]','ビーカー容量 [L]','溶液','ガス交換ユニット材料','流路長 [mm]','回帰係数 a/b/c','備考'],
            '値': [
                mc['date'], mc['recorder'], mc['room_temperature'], mc['water_temperature'],
                "/".join(map(str, mc['ppm_values'])), "/".join(f"{v:.1f}" for v in uatm_vals), mc['ph_concentration'],
                mc['gas_flow'], mc['flow_ml'], mc['beaker_volume'], mc['beaker_solution'],
                mc['gas_unit'], mc['channel_length'], mc['abc'], mc['notes']
            ]
        })
        info_df.to_excel(writer, sheet_name='基本情報', index=False)

                # I/S/R をまとめて1つの縦長テーブルに
        isr_df = pd.DataFrame({
            '値種別': ['I 値', 'S 値', 'R 値'],
            '435 nm': [mc['I_values'][0], mc['S_values'][0], mc['R_values'][0]],
            '490 nm': [mc['I_values'][1], mc['S_values'][1], mc['R_values'][1]],
            '590 nm': [mc['I_values'][2], mc['S_values'][2], mc['R_values'][2]],
            '735 nm': [mc['I_values'][3], mc['S_values'][3], mc['R_values'][3]],
        })
        isr_df.to_excel(writer, sheet_name='I_S_R', index=False)
    st.download_button(
        label='Excelダウンロード',
        data=output.getvalue(),
        file_name=f"測定条件管理_{mc['date']}_{mc['channel_length']}mm.xlsx",
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
