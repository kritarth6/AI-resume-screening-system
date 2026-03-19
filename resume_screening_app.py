import streamlit as st
import joblib
import pandas as pd

# Page config
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🚀",
    layout="wide"
)

# 🔥 ADVANCED CSS (Gradient + Glass UI)
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #1f4037, #99f2c8);
    color: white;
}

/* Glass effect */
.block-container {
    background: rgba(0,0,0,0.6);
    padding: 2rem;
    border-radius: 20px;
}

/* Title styling */
h1 {
    text-align: center;
    font-size: 40px;
    color: #00ffd5;
}

/* Textarea */
textarea {
    background-color: #111 !important;
    color: #00ffd5 !important;
    border-radius: 10px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141e30, #243b55);
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00ffd5, #00c3ff);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px;
}

/* Progress bar color */
div[data-testid="stProgressBar"] > div > div {
    background-color: #00ffd5;
}

/* Hover effect */
.stButton>button:hover {
    transform: scale(1.05);
    transition: 0.3s;
}

</style>
""", unsafe_allow_html=True)

# Load model
model = joblib.load("resume_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Sidebar
st.sidebar.title("🚀 AI Resume Analyzer")
st.sidebar.markdown("""
**👨‍💻 Built by Kritarth Joshi**

✨ NLP + TF-IDF  
✨ Naive Bayes Model  
✨ Real-time Prediction  
""")

# Title
st.title("🚀 AI Resume Screening System")
st.caption("Smart Resume Classification using AI")

st.divider()

# Layout
col1, col2 = st.columns([2,1])

# Input
with col1:
    st.subheader("📄 Enter Resume")
    resume_text = st.text_area(
        "Paste Resume Content",
        height=250,
        placeholder="Paste resume text here..."
    )

# Info
with col2:
    st.subheader("📊 Model Info")
    st.success("Accuracy: ~88%")

    st.markdown("""
    **Model:** Naive Bayes  
    **Technique:** TF-IDF  
    **Dataset:** 962 resumes  
    """)

st.divider()

# Prediction
if resume_text:

    vector = tfidf.transform([resume_text])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]

    st.subheader("🎯 Prediction")

    st.success(f"💼 {prediction}")

    st.divider()

    st.subheader("📊 Confidence")

    categories = model.classes_

    prob_df = pd.DataFrame({
        "Category": categories,
        "Probability": probabilities
    }).sort_values(by="Probability", ascending=False)

    # Progress bars
    for index, row in prob_df.head(5).iterrows():
        st.write(f"**{row['Category']}**")
        st.progress(float(row["Probability"]))

    # Expand details
    with st.expander("📄 Detailed Scores"):
        st.dataframe(prob_df)

# Footer
st.markdown("---")
st.markdown(
    "<center>✨ Built with ❤️ by <b>Kritarth Joshi</b></center>",
    unsafe_allow_html=True
)