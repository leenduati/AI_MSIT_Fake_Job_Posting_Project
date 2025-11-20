import streamlit as st
import joblib
import pandas as pd

# ============================================
# 1. LOAD TRAINED MODEL + VECTORIZER
# ============================================
# These files must exist in the same directory as the Streamlit app
model = joblib.load("model.pkl")        # Trained ML model (Random Forest / Logistic Regression)
vectorizer = joblib.load("vectorizer.pkl")  # TF-IDF vectorizer used during training

# ============================================
# 2. APP TITLE AND INTRO
# ============================================
st.title("Fake Job Postings Detection App")
st.write("Upload a CSV file or paste a job description to predict whether it's REAL or FAKE.")


# ============================================
# 3. SINGLE TEXT PREDICTION
# ============================================
st.header("Single Job Posting Prediction")

# Text input from user
user_input = st.text_area("Enter job title/description/company profile text:")

# Predict button
if st.button("Predict"):

    # Step 3.1: Check if input is empty
    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:
        # Step 3.2: Convert raw text to TF-IDF vector
        vector = vectorizer.transform([user_input])

        # Step 3.3: Model predicts (1 = FAKE, 0 = REAL)
        prediction = model.predict(vector)[0]
        label = "FAKE" if prediction == 1 else "REAL"

        # Step 3.4: Display final prediction
        st.success(f"Prediction: **{label}**")


# ============================================
# 4. BATCH PREDICTION FROM CSV FILE
# ============================================
st.header("Batch Prediction (Upload CSV)")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])

if uploaded_file is not None:

    # Step 4.1: Read CSV into DataFrame
    df = pd.read_csv(uploaded_file)

    # Step 4.2: Detect which column contains the job text
    possible_cols = ["text", "description", "title"]
    text_col = None

    for col in possible_cols:
        if col in df.columns:
            text_col = col
            break

    # Step 4.3: If no valid column found, show error
    if text_col is None:
        st.error("No text column found. Please include 'text', 'title', or 'description' in your CSV.")

    else:
        # Step 4.4: Convert text column to vectors
        vectors = vectorizer.transform(df[text_col].astype(str))

        # Step 4.5: Predict labels for entire file
        df["prediction"] = model.predict(vectors)

        # Step 4.6: Convert numerical labels into readable labels
        df["prediction_label"] = df["prediction"].map({0: "REAL", 1: "FAKE"})

        # Step 4.7: Show preview of results
        st.write(df.head())

        # Step 4.8: Enable download of prediction results
        st.download_button(
            label="Download Predictions",
            data=df.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv"
        )
