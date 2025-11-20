# ============================================
# 1. IMPORT LIBRARIES
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sns.set(style="whitegrid")


# ============================================
# 2. LOAD DATA
# ============================================
df = pd.read_csv("fake_job_postings.csv")

print("Dataset Shape:", df.shape)
print(df.head())


# ============================================
# 3. CLEANING + TEXT PREPROCESSING
# ============================================
df['description'] = df['description'].fillna("")

# Target (y)
df['fraudulent'] = df['fraudulent'].astype(int)

# Combine multiple text fields into one if needed
df['text'] = (
    df['title'].fillna("") + " " +
    df['company_profile'].fillna("") + " " +
    df['description'] + " " +
    df['requirements'].fillna("") + " " +
    df['benefits'].fillna("")
)

print("Text column ready!")


# ============================================
# 4. IMPROVED EXPLORATORY DATA ANALYSIS (EDA)
# ============================================

# 4.1 Distribution of Fake vs Real Jobs
plt.figure(figsize=(6,4))
ax = df['fraudulent'].value_counts().plot(kind='bar')
ax.set_xticklabels(['Real (0)', 'Fake (1)'], rotation=0)
plt.title("Fraudulent vs Real Job Postings")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

print(df['fraudulent'].value_counts(normalize=True))

# 4.2 Missing Values Heatmap
plt.figure(figsize=(14,6))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

# 4.3 Word count distribution
df['word_count'] = df['text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(10,5))
sns.histplot(df['word_count'], kde=True)
plt.title("Distribution of Word Count")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

# 4.4 Compare word count of fake vs real
plt.figure(figsize=(10,5))
sns.boxplot(x='fraudulent', y='word_count', data=df)
plt.title("Word Count Comparison: Fake vs Real Jobs")
plt.xlabel("Fraudulent")
plt.ylabel("Word Count")
plt.show()

# 4.5 Most common words in fake vs real
from collections import Counter
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

fake_text = " ".join(df[df['fraudulent']==1]['clean_text'])
real_text = " ".join(df[df['fraudulent']==0]['clean_text'])

fake_words = Counter(fake_text.split()).most_common(20)
real_words = Counter(real_text.split()).most_common(20)

fake_df = pd.DataFrame(fake_words, columns=['word','count'])
real_df = pd.DataFrame(real_words, columns=['word','count'])

# plot
plt.figure(figsize=(12,5))
sns.barplot(data=fake_df, x='word', y='count')
plt.xticks(rotation=90)
plt.title("Most Common Words in Fake Job Posts")
plt.show()

plt.figure(figsize=(12,5))
sns.barplot(data=real_df, x='word', y='count')
plt.xticks(rotation=90)
plt.title("Most Common Words in Real Job Posts")
plt.show()


# ============================================
# 5. TF-IDF VECTORISATION
# ============================================
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

X = tfidf.fit_transform(df['clean_text'])
y = df['fraudulent']

print("TF-IDF Matrix Shape:", X.shape)


# ============================================
# 6. TRAIN / TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)


# ============================================
# 7. TRAIN MODEL (Logistic Regression)
# ============================================
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)


# ============================================
# 8. EVALUATION
# ============================================
y_pred = model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# ============================================
# 9. SAMPLE PREDICTION FUNCTION
# ============================================
def predict_job(text):
    text = clean_text(text)
    vect = tfidf.transform([text])
    pred = model.predict(vect)[0]
    return "Fake Job Posting" if pred == 1 else "Real Job Posting"

print(predict_job("We are hiring urgently. No experience needed. Send money for registration."))
