# src/evaluate.py

import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay
)
from src.config import MODEL_PATH, PROCESSED_DIR


# ── 1. Load Model ─────────────────────────────────────────────────────────────
def load_model():
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded!")
    return model


# ── 2. Evaluate Model ─────────────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test):


    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # ROC-AUC
    auc = roc_auc_score(y_test, y_pred_prob)
    print(f"\nROC-AUC Score: {auc:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred, y_pred_prob, auc


# ── 3. Plot Confusion Matrix ──────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred):
   
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()
    print("Confusion matrix saved!")


# ── 4. Plot ROC Curve ─────────────────────────────────────────────────────────
def plot_roc_curve(y_test, y_pred_prob):
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig("roc_curve.png")
    plt.show()
    print("ROC curve saved!")


# ── 5. Main Function ──────────────────────────────────────────────────────────
def evaluate():
    

    # Model load karo
    model = load_model()

    # Test data load karo
    with open(os.path.join(PROCESSED_DIR, "X_test.pkl"), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(PROCESSED_DIR, "y_test.pkl"), 'rb') as f:
        y_test = pickle.load(f)

    # Evaluate karo
    y_pred, y_pred_prob, auc = evaluate_model(model, X_test, y_test)

    # Plots
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob)

