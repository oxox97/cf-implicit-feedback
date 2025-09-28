"""
추천 데모 페이지
"""

import streamlit as st
import requests


st.set_page_config(
    page_title="Implicit Feedback CF Demo"
)

st.title("Implicit Feedback CF Demo")

if st.button("모델 학습"):
    response = requests.post("http://localhost:8000/train")

user_id = st.number_input("사용자 ID")
k = st.slider("추천 개수", 5, 20, 10)
if st.button("추천 받기"):
    response = requests.get(f"http://localhost:8000/recommend/{user_id}/{k}")
    data = response.json()
    st.write("추천 결과:", data["recommendations"])