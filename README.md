# Credit Card Fraud Detection

This project implements various machine learning models to detect fraudulent credit card transactions using the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Project Structure

```
├── Notebooks/
│   ├── 01_tree_based_models.ipynb    # Tree-based models (RF, XGBoost, CatBoost)
│   ├── 02_autoencoder_model.ipynb    # Autoencoder for anomaly detection
│   └── 03_summary_analysis.ipynb     # Model comparison and analysis
├── models/                           # Saved model files
├── results/                          # Generated plots and metrics
│   ├── confusion_matrix/
│   ├── f1_threshold/
│   ├── pr_curve/
│   └── roc_curve/
└── requirements.txt                  # Project dependencies
```

## Features

- Multiple model implementations:
  - Tree-based models (Random Forest, XGBoost, CatBoost)
  - Autoencoder for anomaly detection
- Handling of class imbalance using:
  - SMOTE (Synthetic Minority Oversampling)
  - ADASYN (Adaptive Synthetic Sampling)
- Comprehensive evaluation metrics:
  - ROC AUC and PR AUC curves
  - F1-score optimization
  - Precision-Recall analysis
  - Confusion matrices

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the notebooks in order:
   - `01_tree_based_models.ipynb`: Train and evaluate tree-based models
   - `02_autoencoder_model.ipynb`: Train and evaluate the autoencoder
   - `03_summary_analysis.ipynb`: Compare all models and generate visualizations

## Results

The project compares multiple approaches to fraud detection, with a focus on:
- Model performance metrics (F1-score, Precision, Recall)
- Training efficiency
- Impact of different sampling techniques
- Visualization of results through ROC curves and confusion matrices

## Requirements

- Python 3.12+
- See `requirements.txt` for full list of dependencies
