from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# ========================================================================= #
# --- PRODUCTION BALANCED DATASET --- #
df = pd.read_csv("news.csv")
corpus = df["text"]
labels = df["label"]
# ========================================================================= #
# --- SHUFFLE DATA --- #
corpus, labels = shuffle(corpus, labels, random_state=6)

# ========================================================================= #
# --- MODEL PIPELINE --- #
model = Pipeline(
    [
        (
            "vectorizer",
            TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=50000,
                sublinear_tf=True,
                min_df=2,
            ),
        ),
        (
            "classifier",
            LinearSVC(C=1.5,class_weight='balanced', random_state=42),
        ),
    ] 
)
# ========================================================================= #
# --- TRAIN TEST SPLIT --- #
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    corpus, labels, test_size=0.2, random_state=32, stratify=labels
)

# --- TRAIN MODEL --- #
model.fit(X_train, y_train)

# --- PREDICTIONS --- #
predictions = model.predict(X_test)

print(confusion_matrix(y_test, predictions))

# --- ACCURACY --- #
accuracy = accuracy_score(y_test, predictions)
print(classification_report(y_test, predictions))
print(f"Accuracy: {accuracy * 100:.2f}%")
scores = cross_val_score(model, corpus, labels, cv=5, scoring="accuracy", n_jobs=-1)

print(f"All Folds Accuracies: {scores}")
print(f"Mean Cross-Val Accuracy: {scores.mean() * 100:.2f}%")
