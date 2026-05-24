# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import platform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import os

# ── 페이지 기본 설정 ──────────────────────────────────────────────
st.set_page_config(
    page_title="당뇨병 위험도 예측",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 한글 폰트 설정 ────────────────────────────────────────────────
@st.cache_resource
def setup_font():
    """시스템에서 사용 가능한 한글 폰트를 자동으로 찾아 설정합니다."""
    korean_fonts = [
      "Malgun Gothic",
"NanumGothic",
"NanumBarunGothic",
"NanumMyeongjo",
"Apple SD Gothic Neo",
"DejaVu Sans",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in korean_fonts:
        if font in available:
            plt.rc("font", family=font)
            plt.rcParams["axes.unicode_minus"] = False
            return font
    plt.rcParams["axes.unicode_minus"] = False
    return "default"

font_name = setup_font()

# ── 의학 용어 사전 ────────────────────────────────────────────────
MEDICAL_DICTIONARY = {
    "수축기 혈압": "심장이 피를 온몸으로 짜낼 때, 혈관이 받는 '가장 높은 압력'이에요. 흔히 '최고 혈압' 혹은 '앞에 혈압'이라고도 불러요.",
    "이완기 혈압": "심장이 피를 다 짜내고 다시 쉴 때, 혈관에 남은 '가장 낮은 압력'이에요. 흔히 '최저 혈압' 혹은 '뒤에 혈압'이라고 해요.",
    "공복 혈당": "아침에 일어나서 아무것도 먹지 않은 빈속(8시간 이상)일 때 측정한 피 속의 설탕 수치예요. 당뇨를 진단하는 중요한 기준이 됩니다.",
    "당화혈색소": "최근 2~3달 동안 내 몸속의 평균 혈당이 어땠는지 보여주는 성적표예요. 하루 이틀 굶는다고 속일 수 없는 아주 정확한 당뇨 점수랍니다.",
    "체질량지수(BMI)": "내 키에 비해서 몸무게가 적당한지, 혹은 뚱뚱한지를 숫자로 계산해 본 '비만도 성적표'예요.",
    "고혈압": "혈관의 압력이 정상보다 높아서 혈관 벽이 지치고 있는 상태예요. 방치하면 혈관이 막힐 수 있어 약이나 식단으로 꼭 낮춰야 해요.",
    "대사증후군": "배가 많이 나오고, 혈압도 높고, 혈당도 높고, 피에 기름기도 끼는 증상들이 한꺼번에 찾아온 상태예요. 만성질환의 종합선물세트 같아서 관리가 필요해요.",
    "중성지방": "우리가 밥이나 빵을 먹고 남은 에너지가 뱃살이나 핏속에 기름 형태로 저장된 거예요. 너무 많으면 피가 끈적해져서 안 좋아요.",
    "콜레스테롤": "핏속을 돌아다니는 기름기 성분이에요. 좋은 콜레스테롤(HDL)은 혈관을 청소해 주고, 나쁜 콜레스테롤(LDL)은 혈관을 막히게 해요.",
    "당뇨병": "몸속에서 인슐린이라는 열쇠가 제대로 작동하지 않아, 밥을 먹어도 기운으로 가지 못하고 오줌으로 설탕(당)이 섞여 나오는 병이에요.",
}

# ── 데이터 로드 및 모델 학습 (캐싱) ──────────────────────────────
@st.cache_resource
def load_and_train():
    """CSV 파일을 로드하고 모델을 학습합니다. 없으면 데모 데이터를 사용합니다."""

    # CSV 파일 로드 시도
    def try_load(filename):
        for base in [".", "data"]:
            path = os.path.join(base, filename)
            if os.path.exists(path):
                return pd.read_csv(path)
        return None

    df1_raw = try_load("diabetes.csv")
    df2_raw = try_load("diabetes_prediction_dataset.csv")
    df3_raw = try_load("obesity_data.csv")

    dfs = []

    if df1_raw is not None:
        smoking_map_1 = {"Y": 1, "N": 0}
        dfs.append(pd.DataFrame({
            "Age": df1_raw["Age"],
            "Gender": "Female",
            "BMI": df1_raw["BMI"],
            "Glucose": df1_raw["Glucose"],
            "HighBloodPressure": np.where(df1_raw["BloodPressure"] >= 130, 1, 0),
            "Smoking": np.random.choice([0, 1], size=len(df1_raw)),
            "Target_Diabetes": df1_raw["Outcome"],
        }))

    if df2_raw is not None:
        smoking_map_2 = {"never": 0, "No Info": 0, "current": 1, "former": 1, "not current": 0, "ever": 1}
        dfs.append(pd.DataFrame({
            "Age": df2_raw["age"],
            "Gender": df2_raw["gender"],
            "BMI": df2_raw["bmi"],
            "Glucose": df2_raw["blood_glucose_level"],
            "HighBloodPressure": df2_raw["hypertension"],
            "Smoking": df2_raw["smoking_history"].map(smoking_map_2).fillna(0).astype(int),
            "Target_Diabetes": df2_raw["diabetes"],
        }))

    if df3_raw is not None:
        dfs.append(pd.DataFrame({
            "Age": df3_raw["Age"],
            "Gender": df3_raw["Gender"],
            "BMI": df3_raw["BMI"],
            "Glucose": np.nan,
            "HighBloodPressure": np.nan,
            "Smoking": np.nan,
            "Target_Diabetes": np.nan,
        }))

    # 데이터가 없으면 데모 데이터 생성
    if not dfs:
        demo = {
            "Age": np.random.randint(20, 80, 500),
            "Gender": np.random.choice(["Male", "Female"], 500),
            "BMI": np.random.uniform(16, 38, 500),
            "Glucose": np.random.randint(70, 200, 500),
            "HighBloodPressure": np.random.choice([0, 1], 500, p=[0.7, 0.3]),
            "Smoking": np.random.choice([0, 1], 500, p=[0.6, 0.4]),
            "Target_Diabetes": np.random.choice([0, 1], 500, p=[0.8, 0.2]),
        }
        combined = pd.DataFrame(demo)
        using_demo = True
    else:
        combined = pd.concat(dfs, ignore_index=True)
        using_demo = False

    # 데이터 정제
    combined = combined[combined["Gender"].astype(str).str.capitalize().isin(["Male", "Female"])]
    combined["Gender"] = combined["Gender"].astype(str).str.capitalize()

    # 비만도 타겟 생성
    def get_obesity(bmi):
        if bmi < 18.5:   return "저체중"
        elif bmi < 23.0: return "정상 체중"
        elif bmi < 25.0: return "과체중"
        elif bmi < 30.0: return "경도 비만"
        else:            return "고도 비만"

    combined["Target_Obesity"] = combined["BMI"].apply(get_obesity)

    # 결측치 처리
    combined["Glucose"]          = combined["Glucose"].fillna(combined["Glucose"].mean())
    combined["HighBloodPressure"] = combined["HighBloodPressure"].fillna(0)
    combined["Smoking"]           = combined["Smoking"].fillna(0)
    combined["Target_Diabetes"]   = combined["Target_Diabetes"].fillna(0).astype(int)

    # 인코딩
    le_gender  = LabelEncoder()
    le_obesity = LabelEncoder()
    combined["Gender_Encoded"]        = le_gender.fit_transform(combined["Gender"])
    combined["Target_Obesity_Encoded"] = le_obesity.fit_transform(combined["Target_Obesity"])

    # 비만 모델
    X_ob = combined[["Age", "Gender_Encoded", "BMI"]]
    y_ob = combined["Target_Obesity_Encoded"]
    X_tr_ob, X_te_ob, y_tr_ob, y_te_ob = train_test_split(
        X_ob, y_ob, test_size=0.2, random_state=42, stratify=y_ob)
    scaler_ob = StandardScaler()
    X_tr_ob_s = scaler_ob.fit_transform(X_tr_ob)
    X_te_ob_s = scaler_ob.transform(X_te_ob)
    model_obesity = RandomForestClassifier(random_state=42)
    model_obesity.fit(X_tr_ob_s, y_tr_ob)

    # 당뇨 모델
    X_diab = combined[["Age", "Gender_Encoded", "BMI", "Glucose", "HighBloodPressure", "Smoking"]]
    y_diab = combined["Target_Diabetes"]
    X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
        X_diab, y_diab, test_size=0.2, random_state=42, stratify=y_diab)
    scaler_diab = StandardScaler()
    X_tr_d_s = scaler_diab.fit_transform(X_tr_d)
    X_te_d_s = scaler_diab.transform(X_te_d)
    model_diabetes = RandomForestClassifier(random_state=42)
    model_diabetes.fit(X_tr_d_s, y_tr_d)

    # 성능 지표
    rf_diab_pred = model_diabetes.predict(X_te_d_s)
    diab_acc = accuracy_score(y_te_d, rf_diab_pred)
    diab_f1  = f1_score(y_te_d, rf_diab_pred, average="binary")

    return {
        "model_obesity":  model_obesity,
        "model_diabetes": model_diabetes,
        "scaler_ob":      scaler_ob,
        "scaler_diab":    scaler_diab,
        "le_gender":      le_gender,
        "le_obesity":     le_obesity,
        "X_ob":           X_ob,
        "X_diab":         X_diab,
        "diab_acc":       diab_acc,
        "diab_f1":        diab_f1,
        "using_demo":     using_demo,
    }


# ── 비만도 판정 함수 ──────────────────────────────────────────────
def get_obesity_str(bmi: float) -> str:
    if bmi < 18.5:   return "저체중"
    elif bmi < 23.0: return "정상 체중"
    elif bmi < 25.0: return "과체중"
    elif bmi < 30.0: return "경도 비만"
    else:            return "고도 비만"


# ── BMI 게이지 차트 ───────────────────────────────────────────────
def plot_bmi_gauge(bmi: float, name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    segments = [
        (0,    18.5, "#5fa8d3", "저체중"),
        (18.5, 23.0, "#a7c957", "정상"),
        (23.0, 25.0, "#f4a261", "과체중"),
        (25.0, 30.0, "#e76f51", "경도비만"),
        (30.0, 40.0, "#cf1b1b", "비만"),
    ]
    for start, end, color, label in segments:
        ax.barh(["BMI"], [end - start], left=[start], color=color,
                alpha=0.75, label=label, height=0.5)
        ax.text((start + end) / 2, 0, label, ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")

    ax.axvline(x=bmi, color="black", linestyle="--", linewidth=2.5)
    ax.plot(bmi, 0, "ko", markersize=14, zorder=5,
            label=f"내 BMI ({bmi:.1f})")
    ax.set_xlim(10, 42)
    ax.set_xlabel("체질량지수 (BMI)")
    ax.set_title(f"① {name}님의 BMI 위치", fontsize=13, fontweight="bold")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    return fig


# ── 당뇨 위험도 게이지 차트 ──────────────────────────────────────
def plot_diabetes_gauge(prob: float, name: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.barh(["당뇨 위험도"], [40], color="#a7c957", alpha=0.8, label="안전 (0~40%)", height=0.5)
    ax.barh(["당뇨 위험도"], [30], left=[40], color="#ffb703", alpha=0.8, label="주의 (41~70%)", height=0.5)
    ax.barh(["당뇨 위험도"], [30], left=[70], color="#cf1b1b", alpha=0.8, label="위험 (71~100%)", height=0.5)
    for x, lbl in [(20, "안전"), (55, "주의"), (85, "위험")]:
        ax.text(x, 0, lbl, ha="center", va="center",
                fontsize=10, color="white", fontweight="bold")

    ax.axvline(x=prob, color="black", linestyle="-", linewidth=3)
    ax.plot(prob, 0, "y*", markersize=22, markeredgecolor="black", zorder=5,
            label=f"내 위험도 ({prob:.1f}%)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("당뇨 발생 가능성 (%)")
    ax.set_title(f"② {name}님의 당뇨병 위험 게이지", fontsize=13, fontweight="bold")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    return fig


# ── 식생활 가이드 텍스트 ──────────────────────────────────────────
def get_diet_guide(obesity_label: str, diab_prob: float) -> tuple[str, str]:
    # 체중 관리
    if "저체중" in obesity_label:
        weight_advice = (
            "- 근육량 강화를 위해 매끼 단백질(두부, 계란, 살코기, 생선)을 꼭 챙겨 드세요.\n"
            "- 소화가 잘 되도록 조금씩 자주 드시고, 간식으로 우유나 견과류를 권장합니다."
        )
    elif "정상" in obesity_label:
        weight_advice = (
            "- 훌륭한 체중을 유지하고 계십니다! 현재의 균형 잡힌 식습관을 계속 유지해 주세요.\n"
            "- 근손실 방지를 위해 걷기 운동과 콩·두부 섭취를 꾸준히 하시면 좋습니다."
        )
    else:
        weight_advice = (
            "- 무리하게 굶는 다이어트는 건강을 해칩니다. 밥 양을 평소보다 '3분의 1공기'만 줄여보세요.\n"
            "- 국물 요리는 건더기 위주로 드시는 습관이 필요합니다.\n"
            "- 믹스커피·당음료 대신 따뜻한 보리차나 녹차를 드시면 체중 감량에 도움이 됩니다."
        )

    # 당뇨 예방
    diab_advice = """
        ① **식사 순서 바꾸기**: 채소 → 단백질 → 탄수화물 순으로 드셔보세요. 혈당이 천천히 오릅니다.\n
        ② **잡곡밥**: 흰쌀밥·밀가루 음식 대신 현미나 보리, 잡곡을 섞어 드세요.\n
        ③ **규칙적인 식사**: 아침·점심·저녁을 정해진 시간에 드셔야 췌장이 건강합니다.\n
        ④ **식후 산책**: 식사 후 20~30분 뒤 가볍게 걸으시면 혈당이 내려갑니다.
        """
    return weight_advice, diab_advice


# ════════════════════════════════════════════════════════════════
#  메인 UI
# ════════════════════════════════════════════════════════════════
def main():
    # 헤더
    st.markdown(
        """
        <div style='text-align:center; padding: 1.2rem 0 0.5rem 0;'>
            <h1 style='font-size:2.4rem;'>🩺 당뇨병 위험도 예측</h1>
            <p style='color:#666; font-size:1.05rem;'>
                AI 머신러닝 모델이 비만도와 당뇨 위험도를 함께 분석해 드립니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # 모델 로드
    with st.spinner("🔄 AI 모델을 불러오는 중입니다..."):
        m = load_and_train()

    if m["using_demo"]:
        st.info(
            "⚠️ CSV 데이터 파일을 찾지 못해 **데모용 샘플 데이터**로 모델을 학습했습니다.\n"
            "`diabetes.csv`, `diabetes_prediction_dataset.csv` 파일을 같은 폴더에 넣으면 실제 데이터로 동작합니다.",
            icon="ℹ️",
        )

    # ── 사이드바 : 정보 입력 ───────────────────────────────────
    with st.sidebar:
        st.header("📋 건강 정보 입력")
        st.markdown("---")

        name    = st.text_input("이름", value="홍길동", max_chars=20)
        age     = st.slider("나이", min_value=10, max_value=100, value=60, step=1)
        gender  = st.radio("성별", ["Male (남성)", "Female (여성)"], horizontal=True)
        gender_val = "Male" if "Male" in gender else "Female"

        st.markdown("**신체 정보**")
        height = st.slider("키 (cm)", min_value=130, max_value=210, value=165, step=1)
        weight = st.slider("몸무게 (kg)", min_value=30, max_value=150, value=70, step=1)

        st.markdown("**건강 정보**")
        smoking  = st.radio("현재 흡연 중이신가요?", ["아니오", "예"], horizontal=True)
        sbp      = st.slider("최고 혈압 (수축기, mmHg)", 80, 200, 120, 1)
        dbp      = st.slider("최저 혈압 (이완기, mmHg)", 50, 130,  80, 1)
        glucose  = st.slider("공복 혈당 (mg/dL)", 50, 300, 100, 1)

        predict_btn = st.button("🔍 위험도 분석하기", use_container_width=True, type="primary")

    # ── 탭 구성 ───────────────────────────────────────────────
    tab_result, tab_guide, tab_dict, tab_model = st.tabs(
        ["📊 분석 결과", "🥗 건강 가이드", "📖 의학 용어 사전", "🤖 모델 정보"]
    )

    # ── 분석 실행 ─────────────────────────────────────────────
    if predict_btn:
        # 전처리
        height_m        = height / 100
        bmi             = weight / (height_m ** 2)
        high_bp         = 1 if (sbp >= 130 or dbp >= 80) else 0
        smoking_val     = 1 if smoking == "예" else 0
        gender_encoded  = m["le_gender"].transform([gender_val])[0]

        # 비만 예측
        user_ob = pd.DataFrame([[age, gender_encoded, bmi]],
                               columns=["Age", "Gender_Encoded", "BMI"])
        pred_ob_enc   = m["model_obesity"].predict(m["scaler_ob"].transform(user_ob))[0]
        obesity_label = m["le_obesity"].inverse_transform([pred_ob_enc])[0]

        # 당뇨 예측
        user_diab = pd.DataFrame(
            [[age, gender_encoded, bmi, glucose, high_bp, smoking_val]],
            columns=["Age", "Gender_Encoded", "BMI", "Glucose", "HighBloodPressure", "Smoking"],
        )
        pred_diab      = m["model_diabetes"].predict(m["scaler_diab"].transform(user_diab))[0]
        diab_prob      = m["model_diabetes"].predict_proba(m["scaler_diab"].transform(user_diab))[0][1] * 100

        # 체중 가이드
        nw_min = 18.5 * (height_m ** 2)
        nw_max = 23.0 * (height_m ** 2)
        if bmi < 18.5:
            weight_guide = f"정상 범위 도달을 위해 최소 **{nw_min - weight:.1f}kg 증량** 이 필요합니다."
        elif bmi >= 23.0:
            weight_guide = f"정상 범위 도달을 위해 최소 **{weight - nw_max:.1f}kg 감량** 이 필요합니다."
        else:
            weight_guide = f"현재 정상 체중 유지 중입니다. (적정 체중 범위: {nw_min:.1f}kg ~ {nw_max:.1f}kg)"

        # 위험 색상
        if diab_prob <= 40:
            risk_color, risk_text = "#2e7d32", "🟢 안전"
        elif diab_prob <= 70:
            risk_color, risk_text = "#f57f17", "🟡 주의"
        else:
            risk_color, risk_text = "#c62828", "🔴 위험"

        # ─ 탭 1: 분석 결과 ─────────────────────────────────
        with tab_result:
            st.subheader(f"📋 {name}님의 건강 성적표")

            col1, col2, col3 = st.columns(3)
            col1.metric("BMI 점수", f"{bmi:.1f}", obesity_label)
            col2.metric("당뇨 위험도", risk_text, f"{diab_prob:.1f}%")
            col3.metric("혈압 상태", "고혈압 해당" if high_bp else "정상 범위",
                        "⚠️" if high_bp else "✅")

            st.info(f"⚖️ {weight_guide}")
            st.divider()

            st.markdown("#### BMI 게이지")
            st.pyplot(plot_bmi_gauge(bmi, name))

            st.markdown("#### 당뇨 위험도 게이지")
            st.pyplot(plot_diabetes_gauge(diab_prob, name))

        # ─ 탭 2: 건강 가이드 ───────────────────────────────
        with tab_guide:
            weight_advice, diab_advice = get_diet_guide(obesity_label, diab_prob)
            st.subheader(f"🥗 {name}님 맞춤 건강 가이드")
            with st.expander("⚖️ 체중 관리 조언", expanded=True):
                st.markdown(weight_advice)
            with st.expander("🩸 당뇨병 예방·혈당 관리 식습관", expanded=True):
                st.markdown(diab_advice)

            if pred_diab == 1 or diab_prob > 50:
                st.warning(
                    "⚠️ 당뇨 위험도가 높게 나타났습니다. 이 결과는 참고용이며, "
                    "정확한 진단을 위해 반드시 의료 전문가와 상담하시기 바랍니다.",
                    icon="🏥",
                )

        # 세션에 결과 저장
        st.session_state["result_ready"] = True

    else:
        with tab_result:
            st.info("👈 왼쪽 사이드바에서 건강 정보를 입력하고 **'위험도 분석하기'** 버튼을 눌러주세요.")

    # ─ 탭 3: 의학 용어 사전 (항상 표시) ─────────────────────
    with tab_dict:
        st.subheader("📖 의학 용어 돋보기")
        st.caption("병원에서 들은 어려운 의학 용어를 쉽게 풀어드립니다.")
        search = st.text_input("🔍 궁금한 용어를 입력하세요", placeholder="예: 혈당, 콜레스테롤, 당뇨병 …")
        if search:
            found = {k: v for k, v in MEDICAL_DICTIONARY.items() if search in k}
            if found:
                for term, definition in found.items():
                    st.success(f"**💡 [{term}]**\n\n{definition}")
            else:
                st.warning(
                    f"'{search}'에 대한 설명을 찾지 못했어요.\n\n"
                    "**추천 검색어**: 수축기 혈압, 이완기 혈압, 공복 혈당, 당화혈색소, 콜레스테롤, 당뇨병"
                )
        else:
            cols = st.columns(2)
            for i, (term, defn) in enumerate(MEDICAL_DICTIONARY.items()):
                with cols[i % 2]:
                    with st.expander(f"💡 {term}"):
                        st.write(defn)



if __name__ == "__main__":
    main()
