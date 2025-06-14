import numpy as np 
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt



def get_results(y_test, y_pred, y_train, y_pred_train, threshold=0.5, model_name=None):
    """
    Compute and display classification metrics, a classification report, and a confusion matrix for train and test predictions
    using a specified threshold for probability predictions.
    Returns a dictionary with all metrics.
    
    Parameters:
    - y_test: True labels for test set
    - y_pred: Predicted probabilities for test set (e.g., from predict_proba)
    - y_train: True labels for train set
    - y_pred_train: Predicted probabilities for train set (e.g., from predict_proba)
    - threshold: Classification threshold for converting probabilities to binary labels (default: 0.5)
    """
    # Convert probabilities to binary labels based on the threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    y_pred_train_binary = (y_pred_train >= threshold).astype(int)

    # Compute metrics for the test set
    test_accuracy = accuracy_score(y_test, y_pred_binary)
    test_precision = precision_score(y_test, y_pred_binary, zero_division=0)
    test_recall = recall_score(y_test, y_pred_binary, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_binary, zero_division=0)

    # Compute metrics for the training set
    train_accuracy = accuracy_score(y_train, y_pred_train_binary)
    train_precision = precision_score(y_train, y_pred_train_binary, zero_division=0)
    train_recall = recall_score(y_train, y_pred_train_binary, zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train_binary, zero_division=0)

    # Display metrics for the test set
    print("Metrics for test")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"F1: {test_f1:.4f}")

    # Display metrics for the training set
    print("\nMetrics for train")
    print(f"Accuracy: {train_accuracy:.4f}")
    print(f"Precision: {train_precision:.4f}")
    print(f"Recall: {train_recall:.4f}")
    print(f"F1: {train_f1:.4f}")

    # Display classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_binary, zero_division=0))

    # Display confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary, labels=[0, 1])
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
    plt.title(f"Confusion Matrix (Threshold = {threshold:.4f}) for {model_name}")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'../results/confusion_matrix/confusion_matrix_{model_name}_{threshold:.4f}.png', bbox_inches='tight', dpi=150)
    plt.show()

    # Return a dictionary with metrics
    return {
        "test": {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1
        },
        "train": {
            "accuracy": train_accuracy,
            "precision": train_precision,
            "recall": train_recall,
            "f1": train_f1
        }
    }

def plot_roc_curve(model, X_test, y_test, title="ROC Curve"):
    """
    Plot ROC curve for a single model.
    This function computes the ROC curve and AUC for a given model and test dataset,
    and then plots the ROC curve using matplotlib.
    
    Parameters:
    - model: Trained model (e.g., LogisticRegression, XGBClassifier, Keras model)
    - X_test: Test features (pandas DataFrame or numpy array)
    - y_test: True test labels (pandas Series or numpy array)
    - title: Plot title (string, default="ROC Curve")


    Functionality:
    - Computes the ROC curve and AUC using sklearn.metrics functions.
    - Plots the ROC curve using matplotlib, showing the true positive rate vs. false positive rate.
    - Displays the AUC value in the plot and prints it to the console.
    - Returns the AUC value.
    """
    # Get predicted probabilities
    if hasattr(model, 'predict_proba'):  # For sklearn models
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:  # For Keras/TF neural networks
        y_pred_proba = model.predict(X_test, verbose=0).ravel()
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'../results/roc_curve/{title}.png', bbox_inches='tight', dpi=150)
    plt.show()

    # Print AUC
    print(f"AUC: {roc_auc:.4f}")
    return roc_auc

def plot_pr_curve(model, X_test, y_test, title="Precision-Recall Curve"):
    """
    Plot Precision-Recall curve for a single model.

    Parameters:
    - model: Trained model (e.g., LogisticRegression, XGBClassifier, Keras model)
    - X_test: Test features (pandas DataFrame or numpy array)
    - y_test: True test labels (pandas Series or numpy array)
    - title: Plot title (string, default="Precision-Recall Curve")

    Functionality:
    - Computes the precision-recall curve and average precision.
    - Plots the precision-recall curve using matplotlib.
    - Displays the average precision (PR AUC) in the plot and prints it to the console.
    - Returns the average precision (PR AUC) value.
    """

    # Get predicted probabilities
    if hasattr(model, 'predict_proba'):  # For sklearn models
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:  # For Keras/TF models
        y_pred_proba = model.predict(X_test, verbose=0).ravel()

    # Compute Precision-Recall curve and average precision (PR AUC)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.savefig(f'../results/pr_curve/{title}.png', bbox_inches='tight', dpi=150)
    plt.show()

    # Print PR AUC
    print(f"PR AUC: {pr_auc:.4f}")
    return pr_auc


def find_best_f1_threshold(X_test,y_test, model, model_name=None):
    """"
    "Find the best threshold for F1-score by evaluating precision, recall, and F1-score at various thresholds."
    Parameters:
    - X_test: Test features (pandas DataFrame or numpy array)""
    - y_test: True test labels (pandas Series or numpy array)
    - model: Trained model (e.g., LogisticRegression, XGBClassifier, Keras model)
    Functionality:
    - Computes precision, recall, and F1-score at various thresholds.
    - Plots precision, recall, and F1-score against thresholds.
    - Returns the best threshold for F1-score.
    """
    thresholds = np.linspace(0.0, 1.0, 101)
    precisions, recalls, f1s = [], [], []
    y_scores = model.predict_proba(X_test)[:, 1]

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1s.append(f1_score(y_test, y_pred, zero_division=0))

    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    best_f1 = f1s[best_idx]



    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision')
    plt.plot(thresholds, recalls, label='Recall')
    plt.plot(thresholds, f1s, label='F1-score')
    plt.scatter(best_threshold, best_f1, color='red', zorder=5, label=f'Best F1={best_f1:.3f} at {best_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title(f'Precision, Recall, F1-score vs Threshold for {model_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'../results/f1_threshold/f1_threshold_{model_name}.png', bbox_inches='tight', dpi=150)
    plt.show()

    return best_threshold