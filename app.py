import streamlit as st
import numpy as np
import pickle

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="당뇨병 위험도 예측 서비스", layout="centered")

st.title("당뇨병 위험도 예측 서비스")
st.write("건강 정보를 입력하면 AI 모델이 당뇨병 위험도를 예측합니다.")

pregnancies = st.number_input("임신 횟수", 0, 20, 1)
glucose = st.number_input("혈당 수치", 0, 300, 120)
blood_pressure = st.number_input("혈압", 0, 200, 70)
skin_thickness = st.number_input("피부 두께", 0, 100, 20)
insulin = st.number_input("인슐린 수치", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("당뇨 가족력 지표", 0.0, 3.0, 0.5)
age = st.number_input("나이", 1, 120, 30)

if st.button("예측하기"):
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    st.subheader(f"당뇨병 예측 위험도: {probability:.1f}%")
    st.progress(int(probability))

    if prediction == 1:
        st.error("당뇨병 위험이 높은 것으로 예측됩니다.")
    else:
        st.success("당뇨병 위험이 낮은 것으로 예측됩니다.")

    st.caption("※ 본 결과는 학습용 예측 결과이며 실제 의학적 진단이 아닙니다.")