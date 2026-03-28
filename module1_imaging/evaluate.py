from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def evaluate(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_pred))

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def evaluate(y_true, y_scores):
    y_pred = [1 if s > 0.5 else 0 for s in y_scores]

    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_scores))