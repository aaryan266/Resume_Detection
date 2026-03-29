import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset (make sure you have a CSV with 'resume' and 'label')
data = pd.read_csv("resume_dataset.csv")

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

# Apply cleaning
data['resume'] = data['resume'].apply(clean_text)

# Features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['resume'])
y = data['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Test with custom input
def predict_resume(text):
    text = clean_text(text)
    text_vector = vectorizer.transform([text])
    prediction = model.predict(text_vector)
    
    if prediction[0] == 1:
        return "Suitable for job"
    else:
        return "Not suitable"

# Example
sample = "Python developer with machine learning experience and data analysis skills"
print(predict_resume(sample))