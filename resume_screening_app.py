import streamlit as st
import joblib
import pandas as pd
import re

# Page config
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

# ---------- CSS ----------
st.markdown("""
<style>
.stApp {
    background-color: #f4f6f8;
}

.block-container {
    background: white;
    padding: 2rem;
    border-radius: 18px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
    color: #111;
    font-size: 34px;
    font-weight: bold;
}

h2, h3 {
    color: #222;
}

textarea {
    background-color: #ffffff !important;
    color: black !important;
    border-radius: 10px;
}

.stSuccess {
    background-color: #e6f4ea !important;
    color: #1e7e34 !important;
    border-radius: 10px;
}

div[data-testid="stProgressBar"] > div > div {
    background-color: #4caf50;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
model = joblib.load("resume_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# ---------- Title ----------
st.title("📄 AI Resume Screening System")
st.caption("HR-level Resume Analysis using Machine Learning")

st.divider()

# ---------- Input ----------
resume_text = st.text_area(
    "📥 Paste Resume Content",
    height=220,
    placeholder="Paste resume here..."
)

st.divider()

# ---------- Skill Database ----------
skills_db = [
    "python", "machine learning", "deep learning", "tensorflow",
    "keras", "pandas", "numpy", "sql", "power bi",
    "data analysis", "nlp", "computer vision",
    "excel", "tableau", "java", "c++", "react"
]

required_skills = [
    "python", "machine learning", "sql", "data analysis"
]

# ---------- Prediction ----------
if resume_text:

    # ML Prediction
    vector = tfidf.transform([resume_text])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    st.subheader("🎯 Predicted Role")
    st.success(prediction)

    # ---------- Skill Extraction ----------
    resume_lower = resume_text.lower()

    found_skills = [skill for skill in skills_db if skill in resume_lower]
    missing_skills = [skill for skill in required_skills if skill not in resume_lower]

    # ---------- ATS Score ----------
    score = int((len(found_skills) / len(skills_db)) * 100)

    st.subheader("📊 ATS Resume Score")
    st.progress(score / 100)
    st.write(f"**Score: {score}/100**")

    # ---------- Skills Found ----------
    st.subheader("✅ Detected Skills")

    if found_skills:
        st.write(", ".join(found_skills))
    else:
        st.warning("No major skills detected")

    # ---------- Missing Skills ----------
    st.subheader("❌ Missing Important Skills")

    if missing_skills:
        st.write(", ".join(missing_skills))
    else:
        st.success("Great! No critical skills missing")

    st.divider()

    # ---------- Confidence ----------
    st.subheader("📈 Model Confidence")

    categories = model.classes_

    prob_df = pd.DataFrame({
        "Category": categories,
        "Probability": probabilities
    }).sort_values(by="Probability", ascending=False)

    for _, row in prob_df.head(5).iterrows():
        st.write(row["Category"])
        st.progress(float(row["Probability"]))

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<center>✨ Built by Kritarth Joshi | AI Resume Analyzer</center>",
    unsafe_allow_html=True
)