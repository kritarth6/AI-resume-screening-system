import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🚀",
    layout="wide"
)

# 🔥 FORCE FULL BACKGROUND (IMPORTANT FIX)
st.markdown("""
<style>

/* FULL PAGE BACKGROUND */
.stApp {
    background: linear-gradient(135deg, #667eea, #764ba2);
}

/* Remove default black container */
.block-container {
    background: rgba(255, 255, 255, 0.08);
    padding: 2rem;
    border-radius: 20px;
}

/* Title */
h1 {
    text-align: center;
    color: white;
    font-size: 42px;
}

/* Subtitle */
p {
    text-align: center;
    color: #e0e0e0;
}

/* Text area */
textarea {
    background-color: #ffffff !important;
    color: black !important;
    border-radius: 10px;
}

/* Prediction box */
.stSuccess {
    background-color: #00c9a7 !important;
    color: black !important;
    font-weight: bold;
}

/* Progress bar */
div[data-testid="stProgressBar"] > div > div {
    background-color: #ff4b2b;
}

/* Section titles */
h2, h3 {
    color: white;
}

/* Remove sidebar (optional clean look) */
[data-testid="stSidebar"] {
    display: none;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("resume_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Title
st.title("🚀 AI Resume Screening System")
st.write("Smart Resume Classification using AI")

st.divider()

# Input
st.subheader("📄 Enter Resume")

resume_text = st.text_area(
    "Paste Resume Content",
    height=250,
    placeholder="Paste resume text here..."
)

st.divider()

# Prediction
if resume_text:

    vector = tfidf.transform([resume_text])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    st.subheader("🎯 Prediction Result")

    st.success(f"{prediction}")

    st.divider()

    st.subheader("📊 Confidence Scores")

    categories = model.classes_

    prob_df = pd.DataFrame({
        "Category": categories,
        "Probability": probabilities
    }).sort_values(by="Probability", ascending=False)

    for index, row in prob_df.head(5).iterrows():
        st.write(f"**{row['Category']}**")
        st.progress(float(row["Probability"]))

# Footer
st.markdown("---")
st.markdown(
    "<center style='color:white;'>✨ Built by Kritarth Joshi</center>",
    unsafe_allow_html=True
)