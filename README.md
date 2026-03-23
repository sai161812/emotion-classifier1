# Emotion Classifier using NLP

A machine learning project that classifies text into human emotions like **joy, sadness, anger, fear, love, and surprise** using TF-IDF and Logistic Regression.

---

## Overview

This project builds a text classification model that predicts the **emotion behind a given sentence**.
It uses traditional NLP techniques and a lightweight ML model for fast and interpretable results.

---

## Features

* Classifies text into 6 emotions:

  * Sadness
  * Joy
  * Love
  * Anger
  * Fear
  * Surprise
* Text preprocessing (cleaning & normalization)
* TF-IDF vectorization with n-grams
* Efficient Logistic Regression model
* Model saving using `joblib`

---

## Tech Stack

* **Language:** Python
* **Libraries:**

  * pandas
  * scikit-learn
  * joblib
  * regex

---

## Installation

Clone the repository:

```bash
git clone https://github.com/sai161812/emotion-classifier1.git
cd emotion-classifier1
```

Install dependencies:

```bash
pip install pandas scikit-learn joblib
```

---

## ▶Run the Project

```bash
python training.py
```

---

## How It Works

1. Load training and testing datasets
2. Preprocess text (lowercase, remove extra spaces)
3. Convert text into numerical features using **TF-IDF (unigrams + bigrams)**
4. Train a **Logistic Regression classifier**
5. Evaluate using accuracy and classification report
6. Save the trained model as `emotion_classifier.pkl`

---

## Model Details

* **Vectorizer:**

  * n-grams: (1, 2)
  * max features: 50,000
  * min_df: 2

* **Classifier:**

  * Logistic Regression
  * C = 5.0
  * max_iter = 1000
  * class_weight = balanced

---

## Model Export

The trained model is saved as:

```bash
emotion_classifier.pkl
```

You can load it later using:

```python
import joblib
model = joblib.load("emotion_classifier.pkl")
```

---

## Future Improvements

* Try deep learning models (LSTM / BERT)
* Add real-time prediction UI (Streamlit / Flask)
* Improve preprocessing (stopwords, stemming)
* Emotion-Classifier2 will be on another LEVEL

---

## Contributing

Feel free to fork this repo and improve it. Pull requests are welcome.

---
