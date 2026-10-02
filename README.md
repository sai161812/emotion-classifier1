# Emotion classification baseline

A classical NLP baseline for six text labels: sadness, joy, love, anger, fear, and surprise. It uses TF-IDF features and logistic regression; it does not infer a person's actual emotional state.

## Evaluation

The recorded run uses the bundled CSVs, with **15,989 training rows** and **2,000 test rows**. Eleven normalized texts occurred in both files; training removes those 11 rows and preserves the supplied test set. TF-IDF is fitted only on the remaining training data.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority-class baseline (always joy) | 34.75% | 0.0860 |
| TF-IDF + logistic regression | 88.10% | 0.8377 |

[evaluation.json](evaluation.json) records dataset hashes, split counts, per-class metrics, the confusion matrix, error examples, model settings, and library versions. These results describe one supplied test split, not deployment performance.

## What the errors show

| Label | Precision | Recall | F1 | Test examples |
|---|---:|---:|---:|---:|
| Sadness | 0.931 | 0.904 | 0.917 | 581 |
| Joy | 0.915 | 0.901 | 0.908 | 695 |
| Love | 0.707 | 0.836 | 0.767 | 159 |
| Anger | 0.869 | 0.891 | 0.880 | 275 |
| Fear | 0.880 | 0.821 | 0.850 | 224 |
| Surprise | 0.671 | 0.742 | 0.705 | 66 |

The largest confusion is **joy predicted as love: 46 examples**. Love and surprise are weaker than the aggregate accuracy suggests; surprise also has only 66 test examples. Macro-F1 gives each class equal weight, so the two largest classes cannot dominate it as they do accuracy.

## Run

```bash
git clone https://github.com/sai161812/emotion-classifier1.git
cd emotion-classifier1
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python training.py
python usage.py
```

`training.py` writes `emotion_classifier.pkl` and `evaluation.json` alongside the script. `usage.py` applies the same lowercase and whitespace normalization before prediction. Enter a blank line to quit.

The recorded environment is Python 3.12.14, scikit-learn 1.8.0, pandas 2.2.3, and joblib 1.5.3. Use those versions to reproduce the recorded run; the requirements file otherwise permits newer versions.

## Model and limits

- TF-IDF: unigrams and bigrams, up to 50,000 features, `min_df=2`, sublinear term frequency.
- Logistic regression: `C=5.0`, balanced class weights, `lbfgs`, `max_iter=1000`, `random_state=42`. These are the existing model settings; this run does not claim a tuning study.
- The bundled files' source and licence are not documented in the repository.
- Split auditing catches normalized exact matches, not paraphrases or near duplicates. Remaining duplicates within training are retained.
- No cross-validation, external-domain evaluation, deep-learning comparison, or measured inference latency is reported.

Implementation: [training and evaluation](training.py) · [interactive prediction](usage.py).
