# Sentiment Analysis System

### Text Classification using TF-IDF, LinearSVC, and FastAPI

## Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task used to determine the emotional tone of text.
This project builds a machine learning system that automatically classifies text comments into **Positive**, **Negative**, or **Neutral** sentiment.

The system uses **TF-IDF feature extraction**, **Linear Support Vector Classification (LinearSVC)**, and **hyperparameter tuning with RandomizedSearchCV** to achieve improved classification performance.

---

## Objectives

* Perform **text preprocessing and cleaning**
* Convert text data into numerical features using **TF-IDF**
* Train a **machine learning model for sentiment classification**
* Apply **hyperparameter tuning** to improve model performance
* Deploy the model using **FastAPI**

---

## Dataset

* **Dataset:** Sentiment Analysis Dataset
* **Samples Used:** 30,000 text comments
* **Target Classes:**

  * Positive
  * Negative
  * Neutral

**Features**

* Comment (text input)
* Sentiment label

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorizer
* LinearSVC
* RandomizedSearchCV
* FastAPI
* Joblib

---

## Machine Learning Approach

### Text Preprocessing

* Removed missing values
* Cleaned text data
* Sampled 30,000 records
* Split dataset into training and testing sets

### Feature Engineering

Used **TF-IDF Vectorization** with tuned parameters:

* ngram_range
* max_df
* min_df
* max_features

### Model Training

The classification model used:

**Linear Support Vector Classifier (LinearSVC)**

Model hyperparameters were optimized using **RandomizedSearchCV**.

---

## Model Performance

Evaluation metrics used:

* Accuracy
* Precision
* Recall
* F1-Score

**Model Results**

| Metric            | Score |
| ----------------- | ----- |
| Accuracy          | 0.74  |
| Macro F1 Score    | 0.73  |
| Weighted F1 Score | 0.74  |

---

## Example Predictions

| Input Text                        | Predicted Sentiment |
| --------------------------------- | ------------------- |
| "I love this product!"            | Positive            |
| "This is the worst service ever." | Negative            |
| "The product arrived yesterday."  | Neutral             |

---

## API Deployment

The trained model is deployed using **FastAPI**.

### Start the API server

```id="k1x5eg"
uvicorn sentimentapi:app --reload
```

Open API documentation:

```id="s0y7l7"
http://127.0.0.1:8000/docs
```

---

## Example API Request

```json id="wq0bqk"
{
  "text": "This product is amazing"
}
```

### Example API Response

```json id="h6hsz1"
{
  "text": "This product is amazing",
  "prediction": "Positive"
}
```
## Live Demo

https://sentiment-analysis-2-mz2u.onrender.com/docs
---

## Project Structure

```id="8kq7u4"
sentiment-analysis
│
├── api
│   └── app.py
│
├── data
│   └── sentiment_data.csv
│
├── model
│   └── sentiment_model.pkl
│
├── src
│   └── sentiment_pipeline.py
│
├── requirements.txt
└── README.md
```

---

## Conclusion

This project demonstrates how **machine learning and NLP techniques** can be used to analyze text sentiment.
Using **TF-IDF feature engineering, LinearSVC classification, and hyperparameter tuning**, the system effectively predicts sentiment from text data.

