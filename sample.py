# ============================
# 1. Import Required Libraries
# ============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
import joblib

sns.set(style="whitegrid")


# ============================
# 2. Load Dataset
# ============================
df = pd.read_csv("fake_job_postings.csv")
print(df.head())
df.info()
print("\nMissing Values:\n", df.isnull().sum())


# ============================
# 3. Data Cleaning
# ============================
columns_to_drop = [
    'telecommuting', 'has_company_logo', 'has_questions',
    'employment_type', 'required_experience', 'required_education'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Fill missing text fields
text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_columns:
    df[col] = df[col].fillna("")


# ============================
# 4. Exploratory Analysis
# ============================
print("\nClass Distribution:\n", df['fraudulent'].value_counts())

sns.countplot(x='fraudulent', data=df)
plt.title("Distribution of Real vs Fake Job Postings")
plt.show()


# ============================
# 5. Feature Engineering (TF-IDF)
# ============================
df["text"] = (
    df["title"] + " "
    + df["description"] + " "
    + df["company_profile"] + " "
    + df["requirements"] + " "
    + df["benefits"]
)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X = vectorizer.fit_transform(df["text"])
y = df["fraudulent"]


# ============================
# 6. Train-Test Split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training size:", X_train.shape)
print("Test size:", X_test.shape)


# ============================
# 7. Logistic Regression Model
# ============================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)

log_pred = log_model.predict(X_test)
log_proba = log_model.predict_proba(X_test)[:, 1]

print("\n=== Logistic Regression Report ===")
print(classification_report(y_test, log_pred))


# ============================
# 8. Random Forest Model
# ============================
rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"   # improves detection of minority class
)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)[:, 1]

print("\n=== Random Forest Report ===")
print(classification_report(y_test, rf_pred))


# ============================
# 9. Model Evaluation Function
# ============================
def evaluate(y_true, y_pred, y_prob):
    return [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        roc_auc_score(y_true, y_prob)
    ]


log_results = evaluate(y_test, log_pred, log_proba)
rf_results = evaluate(y_test, rf_pred, rf_proba)

comparison_df = pd.DataFrame(
    [log_results, rf_results],
    columns=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
    index=["Logistic Regression", "Random Forest"]
)

print("\n=== Model Performance Comparison ===")
print(comparison_df)


# ============================
# 10. Performance Visualization
# ============================
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, log_results, width, label="Logistic Regression")
plt.bar(x + width/2, rf_results, width, label="Random Forest")
plt.xticks(x, metrics)
plt.ylabel("Score")
plt.title("Logistic Regression vs Random Forest Performance")
plt.legend()
plt.tight_layout()
plt.show()


# ============================
# 11. Save Best Model (Random Forest)
# ============================
joblib.dump(rf_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("\nRandom Forest model and TF-IDF vectorizer saved.")


# ============================
# 12. Sample Prediction Test
# ============================
sample = ["Looking for a data scientist with 3+ years experience in AI and ML"]
sample_vec = vectorizer.transform(sample)
sample_pred = rf_model.predict(sample_vec)

print("\nSample Prediction (0 = Real, 1 = Fake):", sample_pred[0])
