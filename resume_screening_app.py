import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from PyPDF2 import PdfReader
import time

# -------- PAGE CONFIG --------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# -------- CSS --------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b, #020617);
    color: #e2e8f0;
}

.title {
    text-align: center;
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #6366f1, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 30px;
}

.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(14px);
    box-shadow: 0 8px 40px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 20px;
}

.stButton>button {
    background: linear-gradient(90deg, #6366f1, #ec4899);
    color: white;
    border-radius: 12px;
    font-size: 18px;
    padding: 10px 20px;
}

.footer {
    text-align: center;
    margin-top: 40px;
    color: #64748b;
}
</style>
""", unsafe_allow_html=True)

# -------- LOAD MODEL --------
model = joblib.load("resume_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# -------- HEADER --------
st.markdown('<div class="title">🚀 AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload or Paste Resume • Get AI Insights</div>', unsafe_allow_html=True)

# -------- FUNCTIONS --------

def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


def extract_skills(text):
    skills_list = [
        "python", "java", "c++", "machine learning", "deep learning",
        "sql", "excel", "power bi", "tableau", "nlp", "data analysis"
    ]
    return [skill for skill in skills_list if skill in text.lower()]


# ✅ FIXED ATS SCORE (Weighted)
def ats_score(text):
    text = text.lower()

    weights = {
        "skills": 10,
        "experience": 15,
        "project": 15,
        "education": 10,
        "internship": 10,
        "python": 5,
        "sql": 5,
        "machine learning": 10,
        "communication": 5,
        "teamwork": 5,
        "leadership": 5,
        "certification": 5
    }

    score = 0

    for key, value in weights.items():
        if key in text:
            score += value

    return min(score, 100)


# ✅ Missing Keywords (NEW 🔥)
def missing_keywords(text):
    keywords = [
        "python", "sql", "machine learning", "project",
        "internship", "communication", "teamwork", "leadership"
    ]

    text = text.lower()
    return [k for k in keywords if k not in text]


# -------- INPUT SECTION --------
st.markdown('<div class="card">', unsafe_allow_html=True)

option = st.radio("Choose Input Method:", ["Paste Resume", "Upload PDF"])

resume_text = ""

if option == "Paste Resume":
    resume_text = st.text_area("📄 Paste Resume", height=200)

else:
    uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])
    if uploaded_file:
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ PDF Loaded Successfully")

analyze_btn = st.button("🔍 Analyze Resume")

st.markdown('</div>', unsafe_allow_html=True)

# -------- PROCESS --------
if analyze_btn:

    if resume_text.strip() == "":
        st.warning("⚠️ Please provide resume")
    else:

        with st.spinner("🤖 AI is analyzing your resume..."):
            time.sleep(2)

        vector = tfidf.transform([resume_text])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.success(f"🎯 Predicted Role: {prediction}")

        # -------- ATS SCORE --------
        score = ats_score(resume_text)
        st.metric("📊 ATS Score", f"{score}%")

        # -------- SKILLS --------
        skills = extract_skills(resume_text)
        st.write("💡 **Detected Skills:**", skills if skills else "No major skills found")

        # -------- MISSING KEYWORDS --------
        missing = missing_keywords(resume_text)
        st.write("❌ **Missing Keywords (Improve Resume):**", missing if missing else "None 🎉")

        # -------- CHART --------
        categories = model.classes_
        prob_df = pd.DataFrame({
            "Category": categories,
            "Probability": probabilities
        }).sort_values(by="Probability", ascending=False)

        fig = px.bar(
            prob_df.head(8),
            x="Probability",
            y="Category",
            orientation='h',
            text_auto=True,
            title="Confidence Scores"
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # -------- DOWNLOAD REPORT --------
        report = f"""
AI Resume Analysis Report

Predicted Role: {prediction}
ATS Score: {score}%

Skills: {', '.join(skills)}

Missing Keywords: {', '.join(missing)}
        """

        st.download_button("📥 Download Report", report, file_name="resume_report.txt")

# -------- FOOTER --------
st.markdown('<div class="footer">✨ Built by Kritarth Joshi</div>', unsafe_allow_html=True)