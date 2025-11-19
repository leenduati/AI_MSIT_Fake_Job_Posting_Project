# AI Mini-Project: Fake Job Postings Detection
# ============================
# 1. Mount Google Drive
# ============================
# This allows Colab to access files stored in your Google Drive.
# from google.colab import drive
# drive.mount('/content/drive')

# ============================
# 2. Import Required Libraries
# ============================
# pandas & numpy for data handling
# matplotlib & seaborn for visualization
# sklearn for machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ============================
# 3. Load Dataset
# ============================
# Reads the CSV from Google Drive into a pandas DataFrame called df
file_path = 'fake_job_postings.csv'
df = pd.read_csv(file_path)

# Display first 5 rows to inspect the data
print(df.head())

# Check dataset structure and missing values
df.info()
print("\nMissing Values Per Column:\n", df.isnull().sum())

# ============================
# 4. Data Cleaning
# ============================
# Drop irrelevant columns to simplify the dataset
columns_to_drop = ['telecommuting', 'has_company_logo', 'has_questions', 
                   'employment_type', 'required_experience', 'required_education']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Fill missing text fields with empty strings to prevent errors
text_columns = ['title', 'company_profile', 'description', 'requirements', 'benefits']
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna('')

# ============================
# 5. Exploratory Data Analysis (EDA)
# ============================
# Check the balance of real (0) vs fake (1) job postings
print("Class Distribution:\n", df['fraudulent'].value_counts())

# Visualize the distribution
sns.countplot(x='fraudulent', data=df)
plt.title('Distribution of Real vs Fake Job Postings')
plt.show()

# Optional: check job description length distribution
df['description_length'] = df['description'].apply(len)
sns.histplot(df['description_length'], bins=50)
plt.title('Job Description Length Distribution')
plt.show()

# ============================
# 6. Feature Engineering
# ============================
# Combine title, description, and company_profile into one column for text analysis
df['text'] = df['title'] + ' ' + df['description'] + ' ' + df['company_profile']

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['text'])

# Set target variable (0 = Real, 1 = Fake)
y = df['fraudulent']

# ============================
# 7. Train-Test Split
# ============================
# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# 8. Model Training
# ============================
# Initialize Logistic Regression model and train it
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ============================
# Save Model and Vectorizer
# ============================
import joblib

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and vectorizer saved successfully!")


# ============================
# 9. Model Evaluation
# ============================
# Predict labels for the test set
y_pred = model.predict(X_test)

# Accuracy of the model
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc:.4f}")

# Confusion matrix visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
plt.title('Confusion Matrix')
plt.show()

# Detailed classification report
print(classification_report(y_test, y_pred))

# ============================
# 10. Predict on New Job Postings
# ============================
# Example new job posting
new_job_post = ["Looking for a data scientist with 3+ years experience in AI and ML"]

# Transform text to numerical features using the same TF-IDF vectorizer
new_vector = vectorizer.transform(new_job_post)

# Predict if the posting is real (0) or fake (1)
prediction = model.predict(new_vector)
print("Prediction (0 = Real, 1 = Fake):", prediction[0])



