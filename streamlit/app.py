import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import datetime
import lightgbm as lgb
import re
import os

# app.py があるディレクトリを基準にパスを解決（Streamlit Cloud などデプロイ先で cwd が異なる対策）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# モデルファイルのパスマッピング（BASE_DIR からの相対パス）
MODEL_PATHS = {
    '桐生': {
        'course_1': 'モデル/一月用モデル/桐生1_5_78910_56位モデル_0125_0.68.txt',
        'course_6': 'モデル/一月用モデル/桐生6_78910_56位モデル_0125_0.64.txt'
    },
    'びわこ': {
        'course_1': 'モデル/一月用モデル/びわこ1_5_78910_456位モデル_0125_0.79.txt',
        'course_6': 'モデル/一月用モデル/びわこ6_78910_56位モデル_0125_0.63.txt'
    },
    '津': {
        'course_1': 'モデル/一月用モデル/津1_5_78910_456位モデル_0125_0.72.txt',
        'course_6': 'モデル/一月用モデル/津6_78910_456位モデル_0125_0.83.txt'
    },
    '江戸川': {
        'course_1': 'モデル/一月用モデル/江戸川1_5_78910_456位モデル_0125_0.82.txt',
        'course_6': 'モデル/一月用モデル/江戸川6_78910_3456位モデル_0125_0.95.txt'
    },
    '徳山': {
        'course_1': 'モデル/一月用モデル/徳山1_5_78910_56位モデル_0125_0.68.txt',
        'course_6': 'モデル/一月用モデル/徳山6_78910_56位モデル_0125_0.74.txt'
    },
    '下関': {
        'course_1': 'モデル/一月用モデル/下関1_5_78910_56位モデル_0125_0.8.txt',
        'course_6': 'モデル/一月用モデル/下関6_78910_56位モデル_0125_0.53.txt'
    },
    '福岡': {
        'course_1': 'モデル/一月用モデル/福岡1_5_78910_456位モデル_0125_0.76.txt',
        'course_6': 'モデル/一月用モデル/福岡6_78910_456位モデル_0125_0.82.txt'
    }
}


def extract_threshold_from_filename(file_path):
    """
    ファイル名から閾値を抽出する関数
    例: 'びわこ1_5_78910_456位モデル_0125_0.79.txt' -> 0.79
    """
    # ファイル名を取得
    filename = os.path.basename(file_path)
    
    # パターン: _0.XX.txt または _0.XXX.txt の形式を探す
    # 例: _0.79.txt, _0.55.txt, _0.95.txt
    pattern = r'_(\d+\.\d+)\.txt$'
    match = re.search(pattern, filename)
    
    if match:
        threshold = float(match.group(1))
        return threshold
    else:
        # パターンが見つからない場合はデフォルト値を返す
        st.warning(f"⚠️ ファイル名 '{filename}' から閾値を抽出できませんでした。デフォルト値0.5を使用します。")
        return 0.5


def prepare_df(df):
    df = df.copy()

    # ====== 基本整形 ======
    drop_cols = ['名前', 'L数']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # ===== 数値列のクリーニング =====
    clean_numeric_cols = [
        'スタート展示', 'チルト'
    ]

    for col in clean_numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace('\xa0', '', regex=False)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors='coerce')

    #===================


    if '年齢' in df.columns:
        df['年齢'] = df['年齢'].astype(str).str[:2].astype(float)

    if '体重' in df.columns:
        df['体重'] = df['体重'].astype(str).str.replace('/', '', regex=False)
        df['体重'] = pd.to_numeric(df['体重'], errors='coerce')

    if '平均ST' in df.columns:
        df['平均ST'] = df['平均ST'].replace('-', np.nan)
        df['平均ST'] = pd.to_numeric(df['平均ST'], errors='coerce')

    # --- F数の数値化（例: 'F0' → 0, 'F1' → 1） ---
    if 'F数' in df.columns:
        df['F数'] = df['F数'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)

    # --- クラスを数値化 ---
    class_map = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
    df["クラスランク"] = df["クラス"].map(class_map).fillna(0)

    # --- 勝率・連率などの数値化 ---
    rate_columns = [
        '勝率_全国', '勝率_当地', '2連率_全国', '2連率_当地', '3連率_全国', '3連率_当地',
        'モーター2連率', 'モーター3連率', 'ボート2連率', 'ボート3連率'
    ]
    for col in rate_columns:
        if col in df.columns:
            # 文字列の場合、数値に変換（%記号や不要な文字を除去）
            df[col] = df[col].astype(str).str.replace('%', '', regex=False).str.replace(' ', '', regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- コースを数値化 ---
    if 'コース' in df.columns:
        df['コース'] = pd.to_numeric(df['コース'], errors='coerce').fillna(0)

    # --- モーターナンバー、ボートナンバーを数値化 ---
    if 'モーターナンバー' in df.columns:
        df['モーターナンバー'] = pd.to_numeric(df['モーターナンバー'], errors='coerce').fillna(0)
    if 'ボートナンバー' in df.columns:
        df['ボートナンバー'] = pd.to_numeric(df['ボートナンバー'], errors='coerce').fillna(0)

    # ====== 派生特徴量 ======
    df["ST安定スコア"] = 1 / (df["平均ST"] + 0.01)
    df["勝率_diff"] = df["勝率_当地"] - df["勝率_全国"]
    df["2連率_diff"] = df["2連率_当地"] - df["2連率_全国"]
    df["3連率_diff"] = df["3連率_当地"] - df["3連率_全国"]

    df["勝率総合"] = (df["勝率_全国"] + df["勝率_当地"]) / 2
    df["連対安定度"] = (df["2連率_全国"] + df["3連率_全国"]) / 2

    df["モーターパワー"] = (df["モーター2連率"] + df["モーター3連率"]) / 2
    df["ボートパワー"] = (df["ボート2連率"] + df["ボート3連率"]) / 2
    df["機力差"] = df["モーターパワー"] - df["ボートパワー"]
    df["総合機力"] = (df["モーターパワー"] + df["ボートパワー"]) / 2

    course_score = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.4, 5: 0.2, 6: 0.0}
    df["イン有利スコア"] = df["コース"].map(course_score).fillna(0.5)
    # コースが0の場合はNaNを返し、fillna(0)で0に変換
    df["コース逆数"] = (1 / df["コース"].replace(0, np.nan)).fillna(0)

    df["総合力スコア"] = (df["勝率総合"] + df["総合機力"]) / 2
    df["コース適応スコア"] = df["イン有利スコア"] * df["勝率_当地"]
    df["クラス機力スコア"] = df["クラスランク"] * df["総合機力"]
    df["ST勝率連動"] = df["ST安定スコア"] * df["勝率_全国"]
    df["体重ST比"] = df["体重"] / (df["平均ST"] + 0.01)


    # ====== 敵情報特徴量 ======
    df["レースID"] = df["日"].astype(str) + "_" + df["ラウンド"].astype(str)

    agg_features = [
        "勝率_全国", "勝率_当地", "モーターパワー", "ボートパワー",
        "総合力スコア", "クラスランク", "平均ST"
    ]


    # 1. 集計を実行（この時点では列が2層構造）
    race_stats = df.groupby("レースID")[agg_features].agg(['mean', 'max', 'min'])

    # 2. 【重要】MultiIndexを "列名_統計量_全体" に変換
    # c[0]が元の列名、c[1]がmeanやmaxなどの統計量
    race_stats.columns = [f"{c[0]}_{c[1]}_全体" for c in race_stats.columns]
    race_stats = race_stats.reset_index()

    # 3. 元のdfに結合
    df = df.merge(race_stats, on="レースID", how="left")

    # 4. 敵平均の計算（ここでエラーが起きていた箇所）
    for col in ["勝率_全国", "モーターパワー", "総合力スコア"]:
        target_col = f"{col}_mean_全体"
        if target_col in df.columns:
            # 6人レースを想定した計算（(合計 - 自分) / 5人）
            df[f"{col}_敵平均"] = (df[target_col] * 6 - df[col]) / 5
            df[f"{col}_差"] = df[col] - df[f"{col}_敵平均"]

    # クラスランクの敵平均も同様に処理
    if "クラスランク_mean_全体" in df.columns:
        df["敵平均クラスランク"] = (df["クラスランク_mean_全体"] * 6 - df["クラスランク"]) / 5
        df["クラス優位"] = (df["クラスランク"] > df["敵平均クラスランク"]).astype(int)


    # ===== レース内相対 =====
    if 'スタート展示' in df.columns:
        df['スタート展示'] = pd.to_numeric(df['スタート展示'], errors='coerce').fillna(0)
        df['スタート展示_平均との差'] = (
            df['スタート展示']
            - df.groupby('レースID')['スタート展示'].transform('mean')
        )

        df['スタート展示_順位'] = (
            df.groupby('レースID')['スタート展示']
              .rank(method='min')
        )

        df['スタート展示_最速差'] = (
            df['スタート展示']
            - df.groupby('レースID')['スタート展示'].transform('min')
        )

        # ===== コース補正 =====
        df['スタート展示_コース平均との差'] = (
            df['スタート展示']
            - df.groupby('コース')['スタート展示'].transform('mean')
        )

        df['コース_スタート展示'] = df['コース'] * df['スタート展示']

    # ===== チルト =====
    if 'チルト' in df.columns:
        df['チルト'] = pd.to_numeric(df['チルト'], errors='coerce').fillna(0)
        df['チルト_プラス'] = (df['チルト'] > 0).astype(int)
        
        df['チルト_cat'] = df['チルト'].map({
            -0.5: -1,
            0.0:  0,
            0.5:  1
        }).fillna(0)

    # ===== 交互作用 =====
    if 'スタート展示' in df.columns and 'チルト' in df.columns:
        df['スタート展示_チルト'] = df['スタート展示'] * df['チルト']
    if 'チルト' in df.columns and 'コース' in df.columns:
        df['チルト_コース'] = df['チルト'] * df['コース']

    # ====== 目的変数 ======
    # df["1位フラグ"] = df["順位"].isin([5,6]).astype(int)

    # ====== 最終整形 ======
    df.drop(columns=[ "レースID"], inplace=True, errors="ignore")
    df.fillna(0, inplace=True)

    # --- 型変換（object → float） ---
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        except:
            pass

    return df


# ページ設定
st.set_page_config(
    page_title="競艇予測アプリ",
    page_icon="🏁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# カスタムCSS
st.markdown("""
    <style>
    /* メインタイトルのスタイル */
    .main-title {
        text-align: center;
        color: #1f77b4;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    
    /* セクションのスタイル */
    .section-box {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    
    /* カードスタイル */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* ボタンのスタイル改善 */
    .stButton>button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    /* データフレームのスタイル */
    .dataframe {
        border-radius: 8px;
    }
    
    /* 予測結果の強調 */
    .prediction-high {
        background-color: #ff6b6b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .prediction-low {
        background-color: #51cf66;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    /* ヘッダーのスタイル */
    h1, h2, h3 {
        color: #2c3e50;
    }
    
    /* メトリクスカード */
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# 場所のリスト
venues = [
    { 'id': 1, 'name': '桐生' },
    { 'id': 2, 'name': '戸田' },
    { 'id': 3, 'name': '江戸川' },
    { 'id': 4, 'name': '平和島' },
    { 'id': 5, 'name': '多摩川' },
    { 'id': 6, 'name': '浜名湖' },
    { 'id': 7, 'name': '蒲郡' },
    { 'id': 8, 'name': '常滑' },
    { 'id': 9, 'name': '津' },
    { 'id': 10, 'name': '三国' },
    { 'id': 11, 'name': 'びわこ' },
    { 'id': 12, 'name': '住之江' },
    { 'id': 13, 'name': '尼崎' },
    { 'id': 14, 'name': '鳴門' },
    { 'id': 15, 'name': '丸亀' },
    { 'id': 16, 'name': '児嶋' },
    { 'id': 17, 'name': '宮島' },
    { 'id': 18, 'name': '徳山' },
    { 'id': 19, 'name': '下関' },
    { 'id': 20, 'name': '若松' },
    { 'id': 21, 'name': '芦屋' },
    { 'id': 22, 'name': '福岡' },
    { 'id': 23, 'name': '唐津' },
    { 'id': 24, 'name': '大村' },
]

# セッション状態の初期化
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'year' not in st.session_state:
    st.session_state.year = None
if 'month' not in st.session_state:
    st.session_state.month = None
if 'day' not in st.session_state:
    st.session_state.day = None
if 'selected_venue_id' not in st.session_state:
    st.session_state.selected_venue_id = None
if 'selected_venue_name' not in st.session_state:
    st.session_state.selected_venue_name = None
if 'selected_round' not in st.session_state:
    st.session_state.selected_round = None

df = st.session_state.df

# タイトル
st.markdown("""
    <div class="main-title">
        <h1>🏁 競艇予測アプリ</h1>
        <p style="color: #7f8c8d; font-size: 1.1em;">AI予測で着外を予測</p>
    </div>
""", unsafe_allow_html=True)

# メインコンテンツ
st.markdown("---")

# データ取得セクション
with st.container():
    st.markdown("### 📥 レース情報の入力")
    
    # 日付と場所・ラウンドを横並びに
    col1, col2 = st.columns([1, 1])
    
    with col1:
        dt = st.date_input("📅 **日付を選択**", datetime.datetime.today(), key="date_input")
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # スペーサー

# 場所選択
st.markdown("#### 🏟️ 開催場所")
# 6行4列のグリッドを作成
rows = 6
cols = 4
for row in range(rows):
    col_list = st.columns(cols)
    for col_idx in range(cols):
        venue_idx = row * cols + col_idx
        if venue_idx < len(venues):
            venue = venues[venue_idx]
            with col_list[col_idx]:
                # 選択されている場所を色でハイライト
                button_type = "primary" if st.session_state.selected_venue_id == venue['id'] else "secondary"
                
                if st.button(venue['name'], key=f"venue_{venue['id']}", use_container_width=True, type=button_type):
                    st.session_state.selected_venue_id = venue['id']
                    st.session_state.selected_venue_name = venue['name']
                    st.rerun()

# 選択された場所の表示
if st.session_state.selected_venue_id:
    st.success(f"✅ **選択中**: {st.session_state.selected_venue_name}")
else:
    st.info("💡 上記から開催場所を選択してください")

# ラウンド選択
st.markdown("#### 🏁 ラウンド")
# 2行6列のグリッドを作成（12個のボタン）
round_rows = 2
round_cols = 6
for row in range(round_rows):
    col_list = st.columns(round_cols)
    for col_idx in range(round_cols):
        round_num = row * round_cols + col_idx + 1
        if round_num <= 12:
            with col_list[col_idx]:
                # 選択されているラウンドを色でハイライト
                button_type = "primary" if st.session_state.selected_round == round_num else "secondary"
                
                if st.button(f"R{round_num}", key=f"round_{round_num}", use_container_width=True, type=button_type):
                    st.session_state.selected_round = round_num
                    st.rerun()

# 選択されたラウンドの表示
if st.session_state.selected_round:
    st.success(f"✅ **選択中**: R{st.session_state.selected_round}")
else:
    st.info("💡 上記からラウンドを選択してください")

st.markdown("---")

# データ取得ボタン
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🚀 データを取得して予測開始", 
                 disabled=st.session_state.selected_venue_id is None or st.session_state.selected_round is None,
                 use_container_width=True,
                 type="primary"):

        # 日付のフォーマット
        if dt.day < 10:
            day = '0' + str(dt.day)
        else:
            day = str(dt.day)
        
        if dt.month < 10:
            month = '0' + str(dt.month)
        else:
            month = str(dt.month)
        
        year = str(dt.year)
        
        # セッション状態に保存
        st.session_state.year = year
        st.session_state.month = month
        st.session_state.day = day
        
        # 選択された場所のIDを取得
        venue_id = st.session_state.selected_venue_id
        venue_name = st.session_state.selected_venue_name
        
        # 選択されたラウンドを取得
        selected_round = st.session_state.selected_round
        
        # 取得情報をカード形式で表示
        st.markdown("---")
        st.markdown("### 📊 取得情報")
        info_col1, info_col2, info_col3 = st.columns(3)
        with info_col1:
            st.info(f"**日付**: {year}年{month}月{day}日")
        with info_col2:
            st.info(f"**場所**: {venue_name}")
        with info_col3:
            st.info(f"**ラウンド**: R{selected_round}")
        
        # データフレームをリセット
        df = pd.DataFrame()
        
        # プログレスバー
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 選択されたラウンドのみを取得
        r = selected_round
        status_text.text(f"レース {r} のデータを取得中... ({venue_name})")
        
        url = f'https://www.boatrace.jp/owpc/pc/race/racelist?rno={r}&jcd={venue_id:02d}&hd={year}{month}{day}'
        result_url = f'https://www.boatrace.jp/owpc/pc/race/raceresult?rno={r}&jcd={venue_id:02d}&hd={year}{month}{day}'
        
        res = requests.get(url)
        res.encoding = res.apparent_encoding
        soup = BeautifulSoup(res.text, 'html.parser')
        
        p_tag = soup.find('p')
        if p_tag and p_tag.text == '\n知る楽しむ レーサー検索\n':
            # 名前
            name_list_1 = []
            name_list = []
            name = soup.find_all('div', class_="is-fs18")
            name_list_1.append(name)
            for i in name_list_1[0]:
                name_list.append(i.get_text())
            
            # 年齢・体重
            class_age_weight = soup.find_all('div', class_="is-fs11")
            age_weight = []
            x = 1
            for i in class_age_weight:
                if x % 2 == 0:
                    age_weight.append(i.get_text())
                x += 1
            
            age = []
            weight = []
            for i in age_weight:
                age.append(i[30:33])
                weight.append(i[34:38])
            
            x = 1
            class_list = []
            for i in class_age_weight:
                if x % 2 == 1:
                    i = i.get_text()
                    class_list.append(i[35:37])
                x += 1
            
            # データ取得
            data_1 = soup.find_all(class_="is-lineH2")
            
            F_count = []
            L_count = []
            mean_ST = []
            win_rate = []
            win2_rate = []
            win3_rate = []
            local_win_rate = []
            local_win2_rate = []
            local_win3_rate = []
            motor_num = []
            motor_win2 = []
            motor_win3 = []
            boat_num = []
            boat_win2 = []
            boat_win3 = []
            
            x = 0
            for i in range(6):
                F_count.append(data_1[x].get_text()[0:2])
                L_count.append(data_1[x].get_text()[27:29])
                mean_ST.append(data_1[x].get_text()[54:58])
                win_rate.append(data_1[x+1].get_text()[0:4])
                win2_rate.append(data_1[x+1].get_text()[27:34])
                win3_rate.append(data_1[x+1].get_text()[57:64])
                local_win_rate.append(data_1[x+2].get_text()[0:4])
                local_win2_rate.append(data_1[x+2].get_text()[29:34])
                local_win3_rate.append(data_1[x+2].get_text()[57:64])
                motor_num.append(data_1[x+3].get_text()[0:3])
                motor_win2.append(data_1[x+3].get_text()[27:32])
                motor_win3.append(data_1[x+3].get_text()[57:64])
                boat_num.append(data_1[x+4].get_text()[0:3])
                boat_win2.append(data_1[x+4].get_text()[28:33])
                boat_win3.append(data_1[x+4].get_text()[58:63])
                x += 5
            
            course = [1, 2, 3, 4, 5, 6]
            new_df = pd.DataFrame({
                '名前': name_list,
                '年齢': age,
                '体重': weight,
                'クラス': class_list,
                'F数': F_count,
                'L数': L_count,
                '平均ST': mean_ST,
                '勝率_全国': win_rate,
                '2連率_全国': win2_rate,
                '3連率_全国': win3_rate,
                '勝率_当地': local_win_rate,
                '2連率_当地': local_win2_rate,
                '3連率_当地': local_win3_rate,
                'モーターナンバー': motor_num,
                'モーター2連率': motor_win2,
                'モーター3連率': motor_win3,
                'ボートナンバー': boat_num,
                'ボート2連率': boat_win2,
                'ボート3連率': boat_win3,
                'コース': course
            })
            
            time.sleep(1)
            
            # レース結果
            # res = requests.get(result_url)
            # res.encoding = res.apparent_encoding
            # soup = BeautifulSoup(res.text, 'html.parser')
            
            # rank_list = []
            # rank = soup.find_all('td', class_="is-fBold")
            # for i in rank[1:]:
            #     rank_list.append(i.get_text())
            
            # rank_df = pd.DataFrame({
            #     "コース": rank_list,
            #     # "順位": course
            # })
            # rank_df["コース"] = rank_df['コース'].astype(int)
            
            new_df['日'] = f"{day}"
            new_df['ラウンド'] = r

            # 直前情報（スタート展示・チルト）を取得
            status_text.text(f"レース {r} の直前情報を取得中... ({venue_name})")
            info_url = f'https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={r}&jcd={venue_id:02d}&hd={year}{month}{day}'
            
            try:
                info_res = requests.get(info_url)
                info_res.encoding = info_res.apparent_encoding
                info_soup = BeautifulSoup(info_res.text, 'html.parser')
                
                if info_soup.find('p') and info_soup.find('p').text == '\n知る楽しむ レーサー検索\n':
                    # rowspan=4 の td を取得
                    infomation = info_soup.find_all('td', rowspan='4')
                    # 各 td の中の ul を削除
                    for td in infomation:
                        for ul in td.find_all('ul'):
                            ul.decompose()
                    
                    start_list = []
                    tilt_list = []
                    
                    info_num = [3, 10, 17, 24, 31, 38]
                    for info in info_num:
                        if info < len(infomation):
                            start_list.append(infomation[info].text.strip())
                        if info + 1 < len(infomation):
                            tilt_list.append(infomation[info + 1].text.strip())
                    
                    # スタート展示とチルトの情報をデータフレームに追加
                    if len(start_list) == 6 and len(tilt_list) == 6:
                        new_df['スタート展示'] = start_list
                        new_df['チルト'] = tilt_list
                    else:
                        # データが取得できなかった場合は空の値を設定
                        new_df['スタート展示'] = [''] * 6
                        new_df['チルト'] = [''] * 6
                        st.warning(f'レース {r}: 直前情報の取得に問題がありました')
                else:
                    # 直前情報が取得できない場合
                    new_df['スタート展示'] = [''] * 6
                    new_df['チルト'] = [''] * 6
                    st.warning(f'レース {r}: 直前情報がありません')
            except Exception as e:
                # エラーが発生した場合
                new_df['スタート展示'] = [''] * 6
                new_df['チルト'] = [''] * 6
                st.warning(f'レース {r}: 直前情報の取得でエラーが発生しました: {e}')
        
            # new_df = pd.merge(new_df, rank_df, on="コース")
            df = pd.concat([df, new_df], axis=0)
            time.sleep(1)
        else:
            st.warning(f'レース {r}: データがありません')
            
            # プログレスバー更新（1レースのみなので100%）
            progress_bar.progress(1.0)
        
        status_text.text("データ取得完了！")
        st.success("✅ データ取得が完了しました")
        
        # セッション状態に保存
        st.session_state.df = df

# セッション状態から取得
df = st.session_state.df

# メインコンテンツエリア
if not df.empty:
    st.markdown("---")
    
    # 取得したデータの表示（折りたたみ可能）
    with st.expander("📋 取得したデータを表示", expanded=False):
        # コースごとの色を定義
        course_colors = {
            1: '#ffffff',  # 白
            2: '#d3d3d3',  # 灰色
            3: '#ff6b6b',  # 赤色
            4: '#4dabf7',  # 青色
            5: '#ffd43b',  # 黄色
            6: '#51cf66'   # 緑色
        }
        
        # コースごとの色付け関数
        def highlight_by_course(row):
            """コースごとの色を適用"""
            if 'コース' in row.index and pd.notna(row.get('コース')):
                course = int(row['コース'])
                course_color = course_colors.get(course, '#ffffff')
                return [f'background-color: {course_color}; color: #000000'] * len(row)
            return [''] * len(row)
        
        # スタイリングを適用
        if 'コース' in df.columns:
            styled_df = df.style.apply(highlight_by_course, axis=1)
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
    
    test_df = prepare_df(df)
    # 特徴量名取得用に1つモデルを読み込む（起動時は読まない＝デプロイ時のファイル未存在エラーを防ぐ）
    venue_name_for_model = st.session_state.selected_venue_name if st.session_state.selected_venue_name else None
    if venue_name_for_model and venue_name_for_model in MODEL_PATHS:
        _model_path = MODEL_PATHS[venue_name_for_model]['course_1']
    else:
        _model_path = next(iter(MODEL_PATHS.values()))['course_1']
    _model_path_abs = os.path.join(BASE_DIR, _model_path)
    try:
        if os.path.exists(_model_path_abs):
            _bst = lgb.Booster(model_file=_model_path_abs)
            required_columns = _bst.feature_name()
        else:
            required_columns = list(test_df.columns)
    except Exception:
        required_columns = list(test_df.columns)

    # 不足しているカラム
    missing_columns = [col for col in required_columns if col not in test_df.columns]

    # 余分なカラム（test_dfにあるけどrequired_columnsにはないもの）
    extra_columns = [col for col in test_df.columns if col not in required_columns]

    # 表示と処理（エラーがある場合のみ表示）
    if missing_columns:
        with st.expander("⚠️ カラム不足の警告", expanded=False):
            st.write(missing_columns)
        
    for col in missing_columns:
        test_df[col] = np.nan

    if extra_columns:
        test_df = test_df.drop(columns=extra_columns)
    
    cols_to_convert = [
        "勝率_全国", "2連率_全国", "3連率_全国",
        "勝率_当地", "2連率_当地", "3連率_当地",
        "モーターナンバー", "モーター2連率", "モーター3連率",
        "ボートナンバー", "ボート2連率", "ボート3連率",
        "日"
    ]

    for col in cols_to_convert:
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce').astype(float)
    
    # test_df = test_df.drop('4_6位フラグ',axis=1)
    test_df6 = test_df[test_df['コース'] == 6]

    # モデルの読み込み（選択された場所に応じて）
    venue_name = st.session_state.selected_venue_name if st.session_state.selected_venue_name else None
    
    # 使用モデル情報を表示するセクション
    st.markdown("---")
    st.markdown("### 📊 使用モデル情報")
    
    if venue_name and venue_name in MODEL_PATHS:
        try:
            model_info_list = []

                        # コース1のモデルを読み込み
            if not test_df.empty:
                model_path_1 = MODEL_PATHS[venue_name]['course_1']
                model_path_1_abs = os.path.join(BASE_DIR, model_path_1)
                threshold_1 = extract_threshold_from_filename(model_path_1)
                model_filename_1 = os.path.basename(model_path_1)
                bst_1 = lgb.Booster(model_file=model_path_1_abs)
                pred1 = bst_1.predict(test_df)
                test_df['1_5号艇着外予測数値'] = pred1
                pred1_binary = (pred1 > threshold_1).astype(int) 
                test_df['1_5号艇着外予測'] = pred1_binary
                model_info_list.append({
                    'コース': 'コース1',
                    'ファイル名': model_filename_1,
                    '閾値': threshold_1
                })
            
            # コース6のモデルを読み込み
            if not test_df6.empty:
                model_path_6 = MODEL_PATHS[venue_name]['course_6']
                model_path_6_abs = os.path.join(BASE_DIR, model_path_6)
                threshold_6 = extract_threshold_from_filename(model_path_6)
                model_filename_6 = os.path.basename(model_path_6)
                bst_6 = lgb.Booster(model_file=model_path_6_abs)
                pred6 = bst_6.predict(test_df6)
                test_df6['6号艇着外予測数値'] = pred6
                pred6_binary = (pred6 > threshold_6).astype(int) 
                test_df6['6号艇着外予測'] = pred6_binary
                model_info_list.append({
                    'コース': 'コース6',
                    'ファイル名': model_filename_6,
                    '閾値': threshold_6
                })
            

            
            # モデル情報をカード形式で表示
            if model_info_list:
                model_cols = st.columns(len(model_info_list))
                for idx, model_info in enumerate(model_info_list):
                    with model_cols[idx]:
                        st.markdown(f"""
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                        color: white; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                                <h4 style="color: white; margin: 0 0 0.5rem 0;">{model_info['コース']}</h4>
                                <p style="margin: 0.3rem 0; font-size: 0.9em;">閾値: <strong>{model_info['閾値']:.2f}</strong></p>
                                <p style="margin: 0.3rem 0; font-size: 0.8em; opacity: 0.9;">{model_info['ファイル名']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                st.caption("💡 予測値が閾値を超える場合、着外予測が「1」になります")
                st.success(f"✅ {venue_name}のモデルを読み込みました")
        except FileNotFoundError as e:
            st.error(f"⚠️ モデルファイルが見つかりません: {e}")
        except Exception as e:
            st.error(f"⚠️ モデル読み込みエラー: {e}")
    else:
        if venue_name:
            st.warning(f"⚠️ {venue_name}のモデルファイルが登録されていません。予測をスキップします。")
        else:
            st.warning("⚠️ 場所が選択されていません。")

    # URLの生成
    if '日' in test_df.columns and 'ラウンド' in test_df.columns:
        # セッション状態から日付情報を取得
        year = st.session_state.year if st.session_state.year else str(datetime.datetime.today().year)
        month = st.session_state.month if st.session_state.month else f"{datetime.datetime.today().month:02d}"
        # 場所IDを取得（dfに保存されている場合はそれを使用、なければセッション状態から）
        if '場所ID' in test_df.columns:
            venue_id = test_df['場所ID'].iloc[0] if not test_df.empty else st.session_state.selected_venue_id
        else:
            venue_id = st.session_state.selected_venue_id if st.session_state.selected_venue_id else 22
        # 日付情報を元のdfから取得

        test_df = pd.concat([test_df[0:5], test_df6], axis=0)
    
    # 予測結果の表示
    st.markdown("---")
    st.markdown("### 🎯 予測結果")
    
    if not test_df.empty:
        # 表示する列を選択
        display_columns = ['ラウンド', 'コース', 'クラス', '1_5号艇着外予測', '1_5号艇着外予測数値', '6号艇着外予測', '6号艇着外予測数値']
        available_columns = [col for col in display_columns if col in test_df.columns]
        
        if available_columns:
            # データフレームをスタイリング
            styled_df = test_df[available_columns].copy()
            
            # コースごとの色を定義
            course_colors = {
                1: '#ffffff',  # 白
                2: '#d3d3d3',  # 灰色
                3: '#ff6b6b',  # 赤色
                4: '#4dabf7',  # 青色
                5: '#ffd43b',  # 黄色
                6: '#51cf66'   # 緑色
            }
            
            # 予測結果が1の行を判定し、コースごとの色も適用
            def highlight_prediction_1(row):
                """予測結果が1の行をハイライト、コースごとの色も適用"""
                # 1_5号艇着外予測または6号艇着外予測が1の場合
                is_pred_1_5 = False
                is_pred_6 = False
                
                if '1_5号艇着外予測' in row.index and pd.notna(row['1_5号艇着外予測']):
                    is_pred_1_5 = row['1_5号艇着外予測'] == 1
                
                if '6号艇着外予測' in row.index and pd.notna(row['6号艇着外予測']):
                    is_pred_6 = row['6号艇着外予測'] == 1
                
                # コースの色を取得
                course = row.get('コース', 1)
                if pd.notna(course):
                    course = int(course)
                else:
                    course = 1
                course_color = course_colors.get(course, '#ffffff')
                
                if is_pred_1_5 or is_pred_6:
                    # 予測結果が1の場合、オレンジ色の背景（コースの色と組み合わせ）
                    return [f'background-color: #ffa500; color: #000000; font-weight: bold'] * len(row)
                else:
                    # コースごとの色を適用
                    return [f'background-color: {course_color}; color: #000000'] * len(row)
            
            # スタイリングを適用
            styled_df = styled_df.style.apply(highlight_prediction_1, axis=1)
            
            # 予測値に応じて色付け
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400
            )
            
            # 予測サマリー
            st.markdown("#### 📈 予測サマリー")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if '1_5号艇着外予測' in test_df.columns:
                    pred_1_5_count = test_df['1_5号艇着外予測'].sum()
                    st.metric("1-5号艇着外予測", f"{pred_1_5_count}艇", delta=None)
            
            with col2:
                if '6号艇着外予測' in test_df.columns:
                    pred_6_count = test_df['6号艇着外予測'].sum()
                    st.metric("6号艇着外予測", f"{pred_6_count}艇", delta=None)
            
            with col3:
                if '1_5号艇着外予測数値' in test_df.columns:
                    avg_pred_1_5 = test_df['1_5号艇着外予測数値'].mean()
                    st.metric("1-5号艇平均予測値", f"{avg_pred_1_5:.3f}", delta=None)
            
            with col4:
                if '6号艇着外予測数値' in test_df.columns:
                    avg_pred_6 = test_df['6号艇着外予測数値'].mean()
                    st.metric("6号艇平均予測値", f"{avg_pred_6:.3f}", delta=None)

    # URLリンクの表示
    if '日' in test_df.columns and 'ラウンド' in test_df.columns:
        st.markdown("---")
        st.markdown("### 🔗 レース情報")
        year = st.session_state.year if st.session_state.year else str(datetime.datetime.today().year)
        month = st.session_state.month if st.session_state.month else f"{datetime.datetime.today().month:02d}"
        venue_id = st.session_state.selected_venue_id if st.session_state.selected_venue_id else 22
        
        if not test_df.empty:
            round_num = test_df['ラウンド'].iloc[0] if 'ラウンド' in test_df.columns else st.session_state.selected_round
            day = test_df['日'].iloc[0] if '日' in test_df.columns else st.session_state.day
            
            if round_num and day:
                # 数値を整数に変換してから文字列に変換（.0を除去）
                round_num_str = str(int(float(round_num))) if pd.notna(round_num) else str(round_num)
                day_str = str(int(float(day))) if pd.notna(day) else str(day)
                race_url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={round_num_str}&jcd={venue_id:02d}&hd={year}{month}{day_str}"
                st.markdown(f"**[📺 レースページを開く]({race_url})**")
else:
    st.markdown("---")
    st.info("👆 上記から日付・場所・ラウンドを選択して「データを取得して予測開始」ボタンをクリックしてください。")

# フッター
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 2rem 0;">
        <p>🏁 競艇予測アプリ | AI予測システム</p>
    </div>
""", unsafe_allow_html=True)
