# Predicting Airline Recommendation from Customer Reviews

## M508 Big Data Analytics — NLP Project

An end-to-end Natural Language Processing (NLP) pipeline for predicting whether airline customers would recommend an airline based on their written reviews.

## 📌 Project Overview

Airlines receive large volumes of customer reviews, making it difficult to manually identify dissatisfied customers and recurring service issues.

This project develops a text classification system that uses customer review text to predict the existing `Recommended` label.

The project follows an end-to-end NLP workflow:

**Raw Reviews → Text Cleaning & Preprocessing → TF-IDF → Classification → Evaluation → Business Insights**

## 🎯 Business Problem

The objective is to automatically identify reviews that indicate whether a customer would recommend the airline.

The system can support airline customer-experience teams by:

- Identifying potentially dissatisfied customers
- Reducing manual review screening
- Supporting customer-feedback triage
- Highlighting recurring issues in customer feedback
- Supporting data-driven service improvements

## 📊 Dataset

The project uses the **Airline Reviews** dataset by Bhojani (2023), available through Kaggle.

The dataset contains approximately 23,000 airline customer reviews and includes an existing `Recommended` target variable.

The model uses the **review text only** as its input. Numeric sub-ratings were deliberately excluded because they are strongly correlated with the target and could make the classification task largely dependent on numerical thresholds rather than language.

## 🔎 Exploratory Data Analysis

The analysis investigates:

- Dataset structure and size
- Class distribution
- Missing values
- Duplicate reviews
- Review length
- Common words and phrases
- Unigrams and bigrams

The dataset was split using a stratified 80/20 approach:

- **Training reviews:** 18,536
- **Test reviews:** 4,635

## 🧹 Text Preprocessing

The preprocessing pipeline includes:

- Lowercasing
- Punctuation removal
- Stopword removal
- Lemmatization using spaCy

The purpose of preprocessing is to reduce unnecessary textual variation while retaining useful information for classification.

## 🔢 Feature Representation

The cleaned review text is converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

The final representation contains **5,000 features** per review.

Bigrams were also investigated to capture short phrases and contextual word combinations.

## 🤖 Machine Learning Models

Two linear classification approaches were compared:

- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

Both models were trained using the same TF-IDF representation so their performance could be compared consistently.

## 📈 Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Error analysis

Accuracy was not considered sufficient on its own because the dataset contains class imbalance.

## 🔍 Error Analysis

The error analysis examined misclassified reviews to understand where the NLP system struggles.

Important challenges identified include:

- Mixed sentiment
- Context-dependent language
- Negation
- Reviews containing both positive and negative experiences

A key limitation is that removing negation words during preprocessing can change the meaning of expressions such as negative statements.

## 💡 Business Implications

The classifier can potentially be used as a customer-feedback triage tool to automatically identify reviews that require further attention.

However, the analysis indicates that the model should support human review rather than completely replace it.

Potential future improvements include:

- Preserving important negation words
- Confidence-based review
- Classification threshold tuning
- Evaluating context-aware models such as BERT

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- spaCy
- Matplotlib
- Seaborn
- Jupyter Notebook
- TF-IDF
- Logistic Regression
- Linear SVM
