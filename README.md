Fake Job Postings Detection – README
Overview

This project builds a machine learning model that can detect whether a job posting is real or fake using Natural Language Processing (NLP). The workflow involves data loading, cleaning, exploratory analysis, text preprocessing with TF-IDF, model training, evaluation, and prediction on new job postings.

🚀 1. Import Libraries

The code starts by importing the essential Python libraries:

pandas and numpy for data manipulation

matplotlib and seaborn for visualizing data

scikit-learn for machine learning tasks such as TF-IDF, model training, and evaluation

These form the core tools needed for a text classification project.

📥 2. Load Dataset
df = pd.read_csv('fake_job_postings.csv')


This loads the job postings dataset into a pandas DataFrame called df.

The script then displays:

sample rows

data types of each column

missing values

This helps you understand the dataset’s structure before preprocessing.

🧹 3. Data Cleaning

Real-world data is often messy. This section prepares the dataset by:

a) Removing irrelevant columns
df.drop(columns_to_drop, ...)


These fields don’t add value to detecting fake jobs, so removing them simplifies the model.

b) Filling missing text values
df[col] = df[col].fillna('')


Missing descriptions, titles, or profiles are replaced with empty strings to avoid errors in text processing.

📊 4. Exploratory Data Analysis (EDA)

This section helps you understand the dataset visually.

a) Class distribution
df['fraudulent'].value_counts()


Shows how many postings are real (0) vs fake (1).

b) Countplot

A bar chart that shows whether the two classes are balanced.

c) Description length histogram

Calculates and visualizes how long each job description is. Patterns such as:

fake job descriptions may be shorter

real ones tend to be longer

become more visible.

🧪 5. Feature Engineering
Combining multiple text fields
df['text'] = df['title'] + ' ' + df['description'] + ' ' + df['company_profile']


The model works best with one combined text field, so the important text columns are merged.

Converting text into numeric features (TF-IDF)
vectorizer = TfidfVectorizer(...)
X = vectorizer.fit_transform(df['text'])


TF-IDF assigns numerical importance to words.
For example:

Uncommon but meaningful words (like “payment required”) may indicate fraud

Very common words (like “requirements”) contribute less

This creates the machine-readable features used for training.

🎯 6. Prepare the Target Variable
y = df['fraudulent']


y contains the labels:

0 → real job posting

1 → fake job posting

🔀 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(...)


The dataset is divided into:

Training set (80%) – used to train the model

Testing set (20%) – used to measure performance

Using stratify=y ensures both sets contain a similar proportion of real and fake jobs.

🧠 8. Model Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


A Logistic Regression model learns patterns in the text that distinguish real jobs from fake ones.

📈 9. Model Evaluation
a) Predictions
y_pred = model.predict(X_test)

b) Accuracy
accuracy_score(y_test, y_pred)


Shows how many predictions were correct.

c) Confusion Matrix

A heatmap that reveals:

How many real jobs were correctly identified

How many fake jobs were misclassified

d) Classification Report

Includes:

precision

recall

F1-score

These metrics show how reliable the model is at detecting fake job postings.

📝 10. Predicting on New Job Postings

To classify a new job description:

new_job_post = ["Looking for a data scientist..."]
new_vector = vectorizer.transform(new_job_post)
prediction = model.predict(new_vector)


Output:

0 → real job posting

1 → fake job posting

This demonstrates how the model would be used in real-world applications.

✅ Summary in Simple Words

This project takes a dataset of job postings and trains a machine learning model to tell the difference between real and fake job advertisements. It cleans and analyzes the data, transforms text into TF-IDF features, trains a Logistic Regression model, tests its accuracy, and finally uses that model to classify new job descriptions.

The result is essentially a text-based fraud detection system for job postings.
