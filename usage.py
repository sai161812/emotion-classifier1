"""Interactive predictions using the same normalization as training."""

from pathlib import Path

import joblib

from training import emotions, preprocess


def main():
    model_path = Path(__file__).resolve().parent / "emotion_classifier.pkl"
    if not model_path.exists():
        raise SystemExit("Model not found. Run python training.py first.")
    model = joblib.load(model_path)
    while True:
        try:
            text = input("Enter text to classify (blank to quit): ")
        except (EOFError, KeyboardInterrupt):
            break
        if not text.strip():
            break
        print(emotions[int(model.predict([preprocess(text)])[0])])


if __name__ == "__main__":
    main()
