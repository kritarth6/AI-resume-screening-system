import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

# Load model and vectorizer
model = joblib.load("resume_model.pkl")
tfidf = joblib.load("tfidf.pkl")

# Title
st.title("📄 AI Resume Screening System")
st.write("Automatically classify resumes into job categories using **Natural Language Processing**.")

st.divider()

# Two column layout
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Paste Resume Text")
    
    resume_text = st.text_area(
        "Enter Resume Content",
        height=250,
        placeholder="Paste resume text here..."
    )

    predict_btn = st.button("🔍 Predict Category")

with col2:
    st.subheader("Project Info")

    st.info("""
    **Model:** Multinomial Naive Bayes  
    **Feature Engineering:** TF-IDF  
    **Dataset Size:** 962 resumes  
    **Task:** Resume Category Classification
    """)

st.divider()

# Prediction logic
if predict_btn:

    if resume_text.strip() == "":
        st.warning("Please paste resume text first.")
    else:

        # Transform text
        vector = tfidf.transform([resume_text])

        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)[0]

        st.subheader("Prediction Result")

        st.success(f"Predicted Category: **{prediction}**")

        # Probability chart
        categories = model.classes_

        prob_df = pd.DataFrame({
            "Category": categories,
            "Probability": probabilities
        })

        prob_df = prob_df.sort_values(by="Probability", ascending=False)

        st.subheader("Prediction Confidence")

        fig, ax = plt.subplots()
        ax.barh(prob_df["Category"][:10], prob_df["Probability"][:10])
        ax.set_xlabel("Probability")
        ax.set_title("Top Category Probabilities")

        st.pyplot(fig)

st.divider()

st.caption("Built with ❤️ using Python, NLP, Scikit-learn and Streamlit")