import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="centered"
)

# 🔥 PREMIUM CLEAN CSS
st.markdown("""
<style>

/* Background (soft premium gradient) */
.stApp {
    background: linear-gradient(135deg, #f5f7fa, #e4ecf7);
}

/* Main card */
.block-container {
    background: white;
    padding: 2rem;
    border-radius: 20px;
    box-shadow: 0px 10px 30px rgba(0,0,0,0.08);
}

/* Title */
h1 {
    text-align: center;
    color: #1a1a1a;
    font-size: 38px;
    font-weight: 700;
}

/* Subtitle */
p {
    text-align: center;
    color: #555;
    font-size: 16px;
}

/* Text area */
textarea {
    background-color: #f9f9f9 !important;
    color: #000 !important;
    border-radius: 12px !important;
    border: 1px solid #ddd !important;
}

/* Section headers */
h2, h3 {
    color: #222;
}

/* Prediction box */
.stSuccess {
    background-color: #e6f4ea !important;
    color: #1e7e34 !important;
    border-radius: 10px;
    font-weight: 600;
}

/* Progress bars */
div[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #4facfe, #00f2fe);
}

/* Remove sidebar */
[data-testid="stSidebar"] {
    display: none;
}

/* Footer */
.footer {
    text-align: center;
    color: #777;
    font-size: 14px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("resume_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Title
st.title("📄 AI Resume Screening System")
st.write("Analyze resumes using Machine Learning & NLP")

st.divider()

# Input
st.subheader("📥 Paste Resume")

resume_text = st.text_area(
    "",
    height=220,
    placeholder="Paste resume content here..."
)

st.divider()

# Prediction
if resume_text:

    vector = tfidf.transform([resume_text])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    st.subheader("🎯 Result")

    st.success(f"{prediction}")

    st.divider()

    st.subheader("📊 Confidence")

    categories = model.classes_

    prob_df = pd.DataFrame({
        "Category": categories,
        "Probability": probabilities
    }).sort_values(by="Probability", ascending=False)

    for _, row in prob_df.head(5).iterrows():
        st.write(f"{row['Category']}")
        st.progress(float(row["Probability"]))

# Footer
st.markdown(
    "<div class='footer'>Built with ❤️ by Kritarth Joshi</div>",
    unsafe_allow_html=True
)