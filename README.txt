# Resume Screening System

## About the Project
This project is a simple Resume Screening System built using machine learning. It helps in classifying resumes as suitable or not suitable for a job based on their content.

## Objective
The aim of this project is to automate the process of resume screening using basic NLP and machine learning techniques.

## Technologies Used
- Python
- Pandas, Numpy
- NLTK
- Scikit-learn

## Dataset
The dataset should contain resumes and their labels (1 = suitable, 0 = not suitable).

## How It Works
1. Resume text is cleaned (lowercase, remove punctuation and stopwords)
2. Text is converted into numerical form using TF-IDF
3. Data is split into training and testing sets
4. Logistic Regression model is trained
5. The model predicts whether a resume is suitable or not

## How to Run
1. Install dependencies:
   pip install pandas numpy nltk scikit-learn

2. Add dataset file:
   resume_dataset.csv

3. Run:
   python main.py

## Output
The model prints accuracy and also allows testing custom resume input.

## Conclusion
This project shows how machine learning can help automate resume screening in recruitment.