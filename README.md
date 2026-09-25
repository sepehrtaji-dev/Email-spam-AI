# Email Spam AI

A **PyTorch MLP** binary classifier for spam detection, trained on the [Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase) (57 email content features).

## Model

| Component | Details |
|-----------|---------|
| Architecture | MLP: 16 → 8 → 1 |
| Loss | BCEWithLogitsLoss |
| Optimizer | Adam (lr=0.001) |
| Batch size | 256 |
| Epochs | 130 |

## Metrics Reported

Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion Matrix

## Setup

```bash
pip install torch pandas scikit-learn matplotlib
```

## Run

```bash
python model.py
```

## Project Structure

```
Email-spam-AI/
├── model.py          # Model + training + full evaluation
├── cleaning.ipynb    # Data exploration & cleaning
└── spambase_csv.csv  # Dataset (57 features + label)
```
