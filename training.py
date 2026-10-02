"""Train and evaluate a TF-IDF emotion-label baseline on the bundled CSVs."""

import argparse
import hashlib
import json
import platform
import re
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

emotions = ["sadness", "joy", "love", "anger", "fear", "surprise"]


def preprocess(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument("--train", type=Path, default=root / "dataset/train.csv")
    parser.add_argument("--test", type=Path, default=root / "dataset/test.csv")
    parser.add_argument("--model", type=Path, default=root / "emotion_classifier.pkl")
    parser.add_argument("--report", type=Path, default=root / "evaluation.json")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    for name, data in [("train", train), ("test", test)]:
        if not {"text", "label"}.issubset(data.columns):
            raise ValueError(f"{name} must contain text and label columns")
        if data.empty or data[["text", "label"]].isna().any().any():
            raise ValueError(f"{name} contains missing values or no rows")
        if not data.text.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{name} text values must be strings")
        if not data.label.isin(range(len(emotions))).all():
            raise ValueError(f"{name} labels must be integers from 0 to 5")
        data["clean"] = data.text.map(preprocess)

    original_train_rows = len(train)
    duplicate_train_rows = int(train.clean.duplicated().sum())
    overlap = train.clean.isin(set(test.clean))
    removed_overlap_rows = int(overlap.sum())
    removed_overlap_texts = int(train.loc[overlap, "clean"].nunique())
    # Preserve the supplied test set; remove exact normalized text matches from training.
    # This checks membership only and does not use test labels to fit or tune the model.
    train = train.loc[~overlap].copy()
    if train.empty or train.label.nunique() < 2:
        raise ValueError("Insufficient training data after removing split overlap")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000,
                                sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(C=5.0, max_iter=1000,
                                   class_weight="balanced", solver="lbfgs",
                                   random_state=42)),
    ])
    model.fit(train.clean, train.label)
    predictions = model.predict(test.clean)
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(train.clean.to_numpy().reshape(-1, 1), train.label)
    baseline_predictions = baseline.predict(test.clean.to_numpy().reshape(-1, 1))
    labels = list(range(len(emotions)))
    metrics = classification_report(test.label, predictions, labels=labels,
                                    target_names=emotions, output_dict=True,
                                    zero_division=0)
    matrix = confusion_matrix(test.label, predictions, labels=labels)
    confusions = sorted(
        [{"actual": emotions[i], "predicted": emotions[j], "count": int(matrix[i, j])}
         for i in labels for j in labels if i != j and matrix[i, j]],
        key=lambda item: (-item["count"], item["actual"], item["predicted"]),
    )
    errors = []
    for row_index, (actual, predicted) in enumerate(zip(test.label, predictions)):
        if actual != predicted:
            errors.append({"test_row": row_index + 2, "text": str(test.iloc[row_index].text),
                           "actual": emotions[int(actual)], "predicted": emotions[int(predicted)]})
        if len(errors) == 8:
            break

    report = {
        "dataset": {
            "train_sha256": hashlib.sha256(args.train.read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(args.test.read_bytes()).hexdigest(),
            "original_train_rows": original_train_rows,
            "original_train_duplicate_text_rows": duplicate_train_rows,
            "removed_overlap_train_rows": removed_overlap_rows,
            "removed_overlap_unique_texts": removed_overlap_texts,
            "effective_train_rows": len(train), "test_rows": len(test),
            "remaining_split_overlap_unique_texts": len(set(train.clean) & set(test.clean)),
            "train_label_counts": {emotions[i]: int((train.label == i).sum()) for i in labels},
            "test_label_counts": {emotions[i]: int((test.label == i).sum()) for i in labels},
            "policy": "Supplied test set unchanged; normalized exact text overlap removed from training. Remaining within-training duplicates retained.",
        },
        "model": {"features": "TF-IDF", "ngram_range": [1, 2], "max_features": 50000,
                  "sublinear_tf": True, "min_df": 2, "classifier": "LogisticRegression",
                  "C": 5.0, "max_iter": 1000, "class_weight": "balanced",
                  "solver": "lbfgs", "random_state": 42,
                  "observed_iterations": model.named_steps["clf"].n_iter_.tolist()},
        "metrics": {"accuracy": float(accuracy_score(test.label, predictions)),
                    "macro_f1": float(f1_score(test.label, predictions, labels=labels,
                                              average="macro", zero_division=0)),
                    "per_class": {name: metrics[name] for name in emotions},
                    "confusion_matrix": matrix.tolist(), "label_order": emotions},
        "majority_baseline": {"strategy": "most_frequent",
                              "predicted_label": emotions[int(baseline_predictions[0])],
                              "accuracy": float(accuracy_score(test.label, baseline_predictions)),
                              "macro_f1": float(f1_score(test.label, baseline_predictions,
                                                        labels=labels, average="macro", zero_division=0))},
        "largest_confusions": confusions[:5], "error_examples": errors,
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__,
                        "pandas": pd.__version__, "joblib": joblib.__version__},
        "limitations": ["Single supplied test split; no cross-validation or external-domain evaluation.",
                        "Dataset source and licence are not documented in this repository.",
                        "Predicts dataset emotion labels; does not establish a person's emotional state.",
                        "Approximate and paraphrased overlap is not audited."]
    }
    args.model.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Train: {len(train)} rows; test: {len(test)} rows; removed overlap: {removed_overlap_rows}")
    print(f"Accuracy: {report['metrics']['accuracy']:.4f}; macro-F1: {report['metrics']['macro_f1']:.4f}")
    print(f"Majority baseline accuracy: {report['majority_baseline']['accuracy']:.4f}")
    print(classification_report(test.label, predictions, labels=labels,
                                target_names=emotions, zero_division=0))
    print(f"Model: {args.model}\nReport: {args.report}")


if __name__ == "__main__":
    main()
