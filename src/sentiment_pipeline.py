import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import LinearSVC
import joblib


data = pd.read_csv("sentiment_data.csv")

df = data.sample(n=30000, random_state=42)


df = df.dropna(subset=["Comment", "Sentiment"]).reset_index(drop=True)

X = df["Comment"]
y = df["Sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("model",  LinearSVC(max_iter=1000,
                         class_weight="balanced"))
])

param_dist = {
    "tfidf__ngram_range": [(1,1), (1,2), (1,3)],
    "tfidf__max_df": [0.9, 0.95, 1.0],
    "tfidf__min_df": [1,2,5],
    "tfidf__max_features": [5000, 10000, 20000],
    "model__C": [0.1, 1, 10, 100],
    "model__loss": ["hinge", "squared_hinge"]
}


random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=20,       
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
joblib.dump(best_model, "sentiment_model.pkl")