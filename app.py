import streamlit as st
import joblib
import pandas as pd

# ============================================
# 1. LOAD TRAINED MODEL + VECTORIZER
# Use caching to avoid reloading on every interaction
# ============================================
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")          # Trained Random Forest model
    vectorizer = joblib.load("vectorizer.pkl") # TF-IDF vectorizer
    return model, vectorizer

model, vectorizer = load_model()

# ============================================
# 2. APP TITLE AND INTRO
# ============================================
st.title("Fake Job Postings Detection App")
st.write("Paste a job posting or upload a CSV file to predict whether postings are REAL or FAKE.")

# ============================================
# 3. SINGLE TEXT PREDICTION
# ============================================
st.header("Single Job Posting Prediction")
user_input = st.text_area("Enter job title/description/company profile text:")

if st.button("Predict Single Posting"):
    if not user_input.strip():
        st.warning("Please enter some text to predict.")
    else:
        vector = vectorizer.transform([user_input])
        prediction = model.predict(vector)[0]
        label = "FAKE" if prediction == 1 else "REAL"
        st.success(f"Prediction: **{label}**")

# ============================================
# 4. BATCH PREDICTION FROM CSV FILE
# ============================================
st.header("Batch Prediction (Upload CSV)")
uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])

if uploaded_file is not None:
    # Read CSV in chunks to reduce memory usage
    chunksize = 500
    results = []

    for chunk in pd.read_csv(uploaded_file, chunksize=chunksize):
        # Detect text column
        text_col = None
        for col in ["text", "description", "title"]:
            if col in chunk.columns:
                text_col = col
                break

        if text_col is None:
            st.error("No text column found. Include 'text', 'title', or 'description'.")
            break

        # Transform and predict
        vectors = vectorizer.transform(chunk[text_col].astype(str))
        chunk["prediction"] = model.predict(vectors)
        chunk["prediction_label"] = chunk["prediction"].map({0: "REAL", 1: "FAKE"})
        results.append(chunk)

    if results:
        final_df = pd.concat(results, ignore_index=True)
        st.write(final_df.head())

        # Download button
        st.download_button(
            label="Download Predictions",
            data=final_df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
