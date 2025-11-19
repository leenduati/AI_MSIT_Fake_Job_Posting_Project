import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fake Job Postings Detector", layout="wide")

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# -----------------------------
# UI
# -----------------------------
st.title("🕵️ Fake Job Postings Detection App")
st.write("Upload a CSV file to classify job postings as **Real** or **Fake**.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    # -----------------------------
    # Step 1: Load User CSV
    # -----------------------------
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(), use_container_width=True)

    # Required columns
    required_cols = ["title", "company_profile", "description"]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns: {missing}")
        st.stop()

    # -----------------------------
    # Step 2: Clean the Data
    # -----------------------------
    for col in required_cols:
        df[col] = df[col].fillna("")

    df["text"] = (
        df["title"] + " " +
        df["description"] + " " +
        df["company_profile"]
    )

    # -----------------------------
    # Step 3: Vectorize Text
    # -----------------------------
    X_new = vectorizer.transform(df["text"])

    # -----------------------------
    # Step 4: Predict
    # -----------------------------
    predictions = model.predict(X_new)

    df["Prediction"] = predictions
    df["Prediction"] = df["Prediction"].map({0: "Real", 1: "Fake"})

    st.subheader("Prediction Results")
    st.dataframe(df[["title", "Prediction"]], use_container_width=True)

    # -----------------------------
    # Step 5: Download Results
    # -----------------------------
    csv_output = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Predictions as CSV",
        data=csv_output,
        file_name="predictions.csv",
        mime="text/csv"
    )

