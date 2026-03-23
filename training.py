import re
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression as lr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]

def preprocess(text): 
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip() 
    return text

train = pd.read_csv("dataset/train.csv")
test  = pd.read_csv("dataset/test.csv")

train["clean"] = train["text"].apply(preprocess)
test["clean"]  = test["text"].apply(preprocess)

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000, sublinear_tf=True, min_df=2)), #coded in single line for better readablity
    ("clf",   lr(C=5.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42)),])

model.fit(train["clean"], train["label"])

preds = model.predict(test["clean"])
print(f"accuracy: {accuracy_score(test['label'], preds):.4f}")
print(classification_report(test["label"], preds, target_names=emotions))

joblib.dump(model, "emotion_classifier.pkl")