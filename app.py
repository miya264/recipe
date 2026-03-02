#＜3/2　00:00　リョーマ　FINAL版＞

import os
from dataclasses import dataclass
from typing import Dict, List, Set, Optional, Tuple

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 設定（CSVパス & カラー）
# =========================
MAIN_COLOR = "#FF634B"  # 指定のオレンジ色


@dataclass(frozen=True)
class CsvPaths:
    recipes: str = "recipes.csv"
    basic_recipes: str = "basic_recipes_fix.csv"
    ingredients: str = "ingredients_allergensplus.csv"
    allergens: str = "allergens.csv"
    age_state: str = "age_state.csv"
    cook_time: str = "cook_time.csv"
    cost: str = "cost.csv"


# =========================
# データ処理関数
# =========================
def parse_id_list(text: str) -> List[int]:
    if pd.isna(text) or str(text).strip() == "":
        return []
    return [int(x.strip()) for x in str(text).split(",") if x.strip().isdigit()]


def parse_allergen_ids(text: str) -> Set[int]:
    return set(parse_id_list(text))


def safe_int(x, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(x)
    except:
        return default


@st.cache_data
def load_csvs(paths: CsvPaths) -> Dict[str, pd.DataFrame]:
    return {
        "recipes": pd.read_csv(paths.recipes),
        "basic_recipes": pd.read_csv(paths.basic_recipes),
        "ingredients": pd.read_csv(paths.ingredients),
        "allergens": pd.read_csv(paths.allergens),
        "age_state": pd.read_csv(paths.age_state),
        "cook_time": pd.read_csv(paths.cook_time),
        "cost": pd.read_csv(paths.cost),
    }


def build_ingredient_maps(
    dfs: Dict[str, pd.DataFrame],
) -> Tuple[Dict[int, str], Dict[int, Set[int]]]:
    ing = dfs["ingredients"].copy()
    ing_id_to_name = dict(zip(ing["ingredients_id"], ing["ingredients"]))
    ing["allergen_id_set"] = ing["allergens_id"].apply(parse_allergen_ids)
    ing_id_to_allergen_set = dict(zip(ing["ingredients_id"], ing["allergen_id_set"]))
    return ing_id_to_name, ing_id_to_allergen_set


def ingredient_text_from_ids(ids: List[int], ing_id_to_name: Dict[int, str]) -> str:
    names = [str(ing_id_to_name.get(iid, "")) for iid in ids]
    return " ".join([n for n in names if n])


def build_recipes_master_table(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    recipes = dfs["recipes"].copy()
    ing_id_to_name, ing_id_to_allergen_set = build_ingredient_maps(dfs)
    recipes["ingredient_ids"] = recipes["ingredients"].apply(parse_id_list)

    def recipe_allergen_set(ingredient_ids: List[int]) -> Set[int]:
        result = set()
        for iid in ingredient_ids:
            result |= ing_id_to_allergen_set.get(iid, set())
        return result

    recipes["allergen_id_set"] = recipes["ingredient_ids"].apply(recipe_allergen_set)
    recipes["ingredient_text"] = recipes["ingredient_ids"].apply(
        lambda ids: ingredient_text_from_ids(ids, ing_id_to_name)
    )
    recipes["age_state_ids"] = recipes["age_state"].apply(parse_id_list)
    age_id_to_label = dict(
        zip(dfs["age_state"]["age_id"], dfs["age_state"]["age_season"])
    )
    recipes["age_label"] = recipes["age_state_ids"].apply(
        lambda ids: " / ".join(
            [age_id_to_label[i] for i in ids if i in age_id_to_label]
        )
    )

    recipes = recipes.merge(
        dfs["cook_time"].rename(
            columns={"cook_time_id": "cook_time", "cook_time_text": "cook_time_label"}
        ),
        on="cook_time",
        how="left",
    )
    cost_map = dfs["cost"][["cost_id", "cost"]].rename(
        columns={"cost_id": "cost", "cost": "cost_label"}
    )
    recipes = recipes.merge(cost_map, on="cost", how="left")
    return recipes


def build_basic_table(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    basic = dfs["basic_recipes"].copy()
    basic = basic.loc[:, ~basic.columns.str.contains(r"^Unnamed")].copy()
    ing_id_to_name, _ = build_ingredient_maps(dfs)
    basic["ingredient_ids"] = basic["basic_ingredients"].apply(parse_id_list)
    basic["ingredient_text"] = basic["ingredient_ids"].apply(
        lambda ids: ingredient_text_from_ids(ids, ing_id_to_name)
    )
    return basic


@st.cache_data
def build_tfidf(recipes_master: pd.DataFrame) -> Tuple[TfidfVectorizer, any]:
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(
        recipes_master["ingredient_text"].fillna("")
    )
    return vectorizer, tfidf_matrix


def recommend_recipes_from_basic(
    recipes_master: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    base_ingredient_text: str,
    similarity_threshold: float,
    excluded_allergen_ids: Set[int],
    selected_age_state_id: Optional[int],
    top_k: int = 3,
) -> pd.DataFrame:
    base_vec = vectorizer.transform([base_ingredient_text])
    sims = cosine_similarity(base_vec, tfidf_matrix).flatten()
    result = recipes_master.copy()
    result["similarity"] = sims

    result = result[result["similarity"] >= similarity_threshold]
    if excluded_allergen_ids:
        result = result[
            ~result["allergen_id_set"].apply(
                lambda s: len(s & excluded_allergen_ids) > 0
            )
        ]
    if selected_age_state_id is not None:
        result = result[
            result["age_state_ids"].apply(lambda ids: selected_age_state_id in ids)
        ]

    return (
        result.sort_values("similarity", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


# =========================
# Main UI
# =========================
def main():
    st.set_page_config(page_title="あかちゃんごはんナビ", layout="wide")

    # タイトル部分
    st.markdown(f"""
        <div style="text-align: center;">
            <h1 style="color: {MAIN_COLOR}; margin-bottom: 0;">あかちゃんごはんナビ</h1>
            <h3 style="color: {MAIN_COLOR}; margin-top: 0;">もぐもぐ〜ぱくぱく期版</h3>
        </div>
    """, unsafe_allow_html=True)

    # キービジュアル
    col_left, col_mid, col_right = st.columns([0.2, 0.6, 0.2])
    with col_mid:
        if os.path.exists("recommend_app_kv.png"):
            st.image("recommend_app_kv.png", use_container_width=True)
        else:
            st.warning("recommend_app_kv.png が見つかりません。")

    # 説明文
    st.markdown(f"""
        <div style='color: #333333; text-align: left; font-size: 18px; line-height: 1.6; font-weight: bold; padding: 15px 0;'>
            あなたがお子さまに作ってあげたい料理から、アレルギーや月齢を考慮したおすすめレシピをご提案します。<br>
            お子さまにぴったりのメニュー選びをお手伝いします。
        </div>
    """, unsafe_allow_html=True)

    # データロード
    paths = CsvPaths()
    dfs = load_csvs(paths)
    recipes_master = build_recipes_master_table(dfs)
    basic_table = build_basic_table(dfs)
    vectorizer, tfidf_matrix = build_tfidf(recipes_master)

    st.divider()

    # 検索セクション
    st.markdown(f"#### <span style='color: {MAIN_COLOR};'>🔍 今日は何を作ってあげようかな</span>", unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown("**① カテゴリー**")
            categories = ["すべて"] + sorted(basic_table["category"].dropna().unique().tolist())
            sel_cat = st.selectbox("カテゴリーを選択", options=categories, key="cat_sel", label_visibility="collapsed")
        
        with c2:
            st.markdown("**② 作ってあげたい料理**")
            if sel_cat == "すべて":
                recipe_options = basic_table["basic_recipe"].tolist()
            else:
                recipe_options = basic_table[basic_table["category"] == sel_cat]["basic_recipe"].tolist()
            
            base_recipe_name = st.selectbox("起点レシピ", options=recipe_options, key="base_recipe", label_visibility="collapsed")
            base_text = basic_table.loc[basic_table["basic_recipe"] == base_recipe_name, "ingredient_text"].iloc[0]
            
        with c3:
            st.markdown("**③ お子さまのアレルギー**")
            all_list = dfs["allergens"]["allergens"].tolist()
            sel_allergens = st.multiselect("アレルギー", options=all_list, key="allergens_sel", label_visibility="collapsed")
            all_map = dict(zip(dfs["allergens"]["allergens"], dfs["allergens"]["id"]))
            excl_ids = {int(all_map[n]) for n in sel_allergens}
            
        with c4:
            st.markdown("**④ お子さまの月齢**")
            # --- 修正：部分一致で「5～6」が含まれる項目を確実に除外 ---
            age_all_list = dfs["age_state"]["age_season"].tolist()
            age_list = [age for age in age_all_list if "5～6" not in str(age)]
            
            sel_age = st.selectbox("月齢", options=["指定なし"] + age_list, key="age_sel", label_visibility="collapsed")
            age_id = None
            if sel_age != "指定なし":
                age_id = int(dfs["age_state"].loc[dfs["age_state"]["age_season"] == sel_age, "age_id"].iloc[0])

        st.markdown(" ")
        st.markdown(f"""
            <style>
                div.stButton > button:first-child {{
                    background-color: {MAIN_COLOR};
                    border-color: {MAIN_COLOR};
                    color: white;
                }}
                div.stButton > button:hover {{
                    background-color: {MAIN_COLOR};
                    border-color: {MAIN_COLOR};
                    opacity: 0.9;
                }}
            </style>
        """, unsafe_allow_html=True)
        search_btn = st.button("この条件でレシピを検索する", use_container_width=True)

    # 結果表示
    if search_btn:
        rec_df = recommend_recipes_from_basic(recipes_master, vectorizer, tfidf_matrix, base_text, 0.05, excl_ids, age_id)

        if rec_df.empty:
            st.error("条件に合うレシピが見つかりませんでした。")
        else:
            st.markdown("---")
            st.markdown(f"### <span style='color: {MAIN_COLOR};'>あなたのお子さまにおすすめのレシピ</span>", unsafe_allow_html=True)
            
            st.markdown(f"""
                <div style='background-color: #FFF5F4; padding: 15px; border-radius: 5px; border-left: 5px solid {MAIN_COLOR}; margin-bottom: 25px;'>
                    <div style='color: #333333; font-size: 16px; font-weight: bold; margin-bottom: 5px;'>
                        💡 これらのレシピが選ばれた理由
                    </div>
                    <div style='color: #555555; font-size: 14px; line-height: 1.5;'>
                        選択された「{base_recipe_name}」と使っている材料が近いため、お子さまの好みに合った味付けや食感に近いレシピとして選出しました。<br>
                        アレルギーや月齢の条件も考慮しています。
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            for i, row in rec_df.iterrows():
                rank = i + 1
                with st.container(border=True):
                    st.markdown(f"#### 🏆 おすすめ第{rank}位!")
                    c_img, c_info = st.columns([1, 2])
                    
                    with c_img:
                        img_url = row.get("image_url")
                        st.image(img_url if isinstance(img_url, str) and img_url.startswith("http") else "https://via.placeholder.com/400x300?text=No+Image", use_container_width=True)
                    
                    with c_info:
                        st.subheader(row.get('title', 'メニュー名'))
                        
                        g_col, t_col = st.columns([1, 1])
                        
                        with g_col:
                            sim_percent = max(0, min(100, int(float(row.get('similarity', 0)) * 100)))
                            remaining = 100 - sim_percent
                            
                            fig = go.Figure(data=[go.Pie(
                                values=[sim_percent, remaining],
                                hole=.75,
                                marker_colors=[MAIN_COLOR, '#EEEEEE'],
                                textinfo='none',
                                hoverinfo='none',
                                sort=False,
                                direction='clockwise',
                                rotation=0
                            )])

                            fig.update_layout(
                                showlegend=False,
                                margin=dict(t=0, b=0, l=0, r=0),
                                height=160,
                                annotations=[dict(
                                    text=f'<b>{sim_percent}%</b>', 
                                    x=0.5, y=0.5, 
                                    font_size=24, 
                                    showarrow=False, 
                                    font_family="Arial Black",
                                    font_color=MAIN_COLOR
                                )]
                            )
                            st.write("**★作ってあげたい料理とのマッチ度**")
                            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, key=f"chart_{rank}")

                        with t_col:
                            age_label = row.get('age_label')
                            age_display = age_label if pd.notna(age_label) else "指定なし/不明"
                            st.markdown(f"**★対象の月齢**")
                            st.write(f"👶 {age_display}")
                            st.markdown(f"**★費用**")
                            st.write(f"💰 {row.get('cost_label', '不明')}")
                            st.markdown(f"**★調理時間**")
                            st.write(f"⏱ {row.get('cook_time_label', '不明')}")
                        
                        st.divider()
                        
                        recipe_url = row.get("url")
                        if recipe_url and isinstance(recipe_url, str) and recipe_url.startswith("http"):
                            st.link_button("✨ レシピ詳細を確認する (外部サイト)", recipe_url, use_container_width=True)
                        else:
                            st.button(f"レシピ詳細（準備中）-{rank}", disabled=True, use_container_width=True)
    else:
        st.markdown("---")
        st.markdown("<center>上記で条件を選んで「検索する」ボタンを押してください</center>", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
        <div style='background-color: #f9f9f9; padding: 15px; border-radius: 5px; color: #666666; font-size: 12px; line-height: 1.6;'>
            <b>【注意事項】</b><br>
            本アプリは情報の提供を目的としたものであり、特定の診断を保証するものではありません。
            レシピを利用する際は医師の診断・指導にもとづいて原因食物を確認し、適切な食材を選択してください。
            加工食品を使用する際は、必ずご自身で最新の原材料やアレルギー表示をご確認ください。
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()