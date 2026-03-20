import streamlit as st
import joblib
import pandas as pd
import plotly.express as px

# Page Config
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")

# -------- PREMIUM UI CSS --------
st.markdown("""
<style>

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #1e3c72, #2a5298, #4facfe);
    color: white;
}

/* Main Title */
.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: white;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #dbeafe;
    margin-bottom: 30px;
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.2);
}

/* Text Area */
textarea {
    background-color: rgba(255,255,255,0.1) !important;
    color: white !important;
    border-radius: 10px !important;
}

/* Button */
.stButton>button {
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    border-radius: 10px;
    font-size: 18px;
    font-weight: bold;
    border: none;
    padding: 10px 20px;
}

/* Footer */
.footer {
    text-align: center;
    margin-top: 40px;
    color: #e2e8f0;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# -------- LOAD MODEL --------
model = joblib.load("resume_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# -------- HEADER --------
st.markdown('<div class="title">🚀 AI Resume Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart Resume Classification using AI & NLP</div>', unsafe_allow_html=True)

# -------- INPUT CARD --------
st.markdown('<div class="card">', unsafe_allow_html=True)

resume_text = st.text_area("📄 Paste Resume Content", height=250)

predict_btn = st.button("🔍 Analyze Resume")

st.markdown('</div>', unsafe_allow_html=True)

# -------- PREDICTION --------
if predict_btn:

    if resume_text.strip() == "":
        st.warning("⚠️ Please enter resume text")
    else:
        vector = tfidf.transform([resume_text])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.success(f"🎯 Predicted Category: {prediction}")

        # Probability Chart
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
            title="Confidence Scores",
            text_auto=True
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# -------- FOOTER --------
st.markdown('<div class="footer">✨ Built by Kritarth Joshi | AI Resume Analyzer</div>', unsafe_allow_html=True)