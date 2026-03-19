import streamlit as st
import joblib
import pandas as pd
import re

# ---------- CONFIG ----------
st.set_page_config(page_title="AI Resume Analyzer", page_icon="🚀", layout="centered")

# ---------- DARK PREMIUM CSS ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #1e293b, #0f172a);
}

/* Card */
.block-container {
    background: #111827;
    padding: 2rem;
    border-radius: 18px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.6);
}

/* Title */
h1 {
    text-align: center;
    color: #f8fafc !important;
}

/* Text */
p {
    color: #cbd5e1 !important;
    text-align: center;
}

/* Input */
textarea {
    background-color: #020617 !important;
    color: #f1f5f9 !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
}

/* Result box */
.stSuccess {
    background-color: #022c22 !important;
    color: #4ade80 !important;
}

/* Progress */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #38bdf8, #22c55e);
}

/* Skill tags */
.tag {
    display: inline-block;
    background: #1e293b;
    color: #38bdf8;
    padding: 6px 10px;
    margin: 5px;
    border-radius: 8px;
    font-size: 13px;
}

/* Missing skills */
.missing {
    background: #3f1d1d;
    color: #f87171;
}

/* Remove sidebar */
[data-testid="stSidebar"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD ----------
model = joblib.load("resume_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# ---------- TITLE ----------
st.title("🚀 AI Resume Analyzer")
st.caption("Smart Resume Screening with AI (HR-Level Tool)")

st.divider()

# ---------- INPUT ----------
resume_text = st.text_area("📥 Paste Resume", height=220)

# ---------- SKILL DB ----------
skills_db = [
    "python","machine learning","deep learning","tensorflow","keras",
    "pandas","numpy","sql","power bi","nlp","computer vision",
    "excel","tableau","java","c++","react"
]

required_skills = ["python","machine learning","sql","data analysis"]

# ---------- FUNCTION: HIGHLIGHT ----------
def highlight_skills(text, skills):
    for skill in skills:
        text = re.sub(f"({skill})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
    return text

# ---------- MAIN ----------
if resume_text:

    # ML Prediction
    vector = tfidf.transform([resume_text])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    st.subheader("🎯 Predicted Role")
    st.success(prediction)

    # ---------- SKILL DETECTION ----------
    resume_lower = resume_text.lower()

    found_skills = [s for s in skills_db if s in resume_lower]
    missing_skills = [s for s in required_skills if s not in resume_lower]

    # ---------- ATS SCORE ----------
    score = int((len(found_skills) / len(skills_db)) * 100)

    st.subheader("📊 ATS Score")
    st.progress(score / 100)
    st.write(f"### {score}/100")

    # ---------- SKILLS ----------
    st.subheader("✅ Detected Skills")

    if found_skills:
        for skill in found_skills:
            st.markdown(f"<span class='tag'>{skill}</span>", unsafe_allow_html=True)
    else:
        st.warning("No major skills detected")

    # ---------- MISSING ----------
    st.subheader("❌ Missing Skills")

    if missing_skills:
        for skill in missing_skills:
            st.markdown(f"<span class='tag missing'>{skill}</span>", unsafe_allow_html=True)
    else:
        st.success("No critical skills missing")

    st.divider()

    # ---------- HIGHLIGHT ----------
    st.subheader("🔍 Resume Analysis (Highlighted Skills)")
    highlighted = highlight_skills(resume_text, found_skills)
    st.markdown(highlighted, unsafe_allow_html=True)

    st.divider()

    # ---------- CONFIDENCE ----------
    st.subheader("📈 Model Confidence")

    categories = model.classes_

    prob_df = pd.DataFrame({
        "Category": categories,
        "Probability": probabilities
    }).sort_values(by="Probability", ascending=False)

    for _, row in prob_df.head(5).iterrows():
        st.write(row["Category"])
        st.progress(float(row["Probability"]))

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<center style='color:#94a3b8;'>✨ Built by Kritarth Joshi | AI Resume Analyzer</center>",
    unsafe_allow_html=True
)