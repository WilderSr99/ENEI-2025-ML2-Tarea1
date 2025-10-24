from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def report_classification(y_true, y_pred, title="Clasificación"):
    print(f"\n--- {title} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"F1-score: {f1_score(y_true, y_pred):.3f}")
    print(classification_report(y_true, y_pred))
    
def plot_confusion(y_true, y_pred, labels=None, title="Matriz de confusión"):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicción")
    plt.ylabel("Verdadero")
    plt.show()
