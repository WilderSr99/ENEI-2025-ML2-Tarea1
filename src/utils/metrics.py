"""
metrics.py — Funciones para evaluación de modelos (F1, accuracy, matriz de confusión).
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report


def report_classification(y_true, y_pred, title="Clasificación"):
    """
    Imprime accuracy, F1-score y reporte de clasificación.
    """
    print(f"\n--- {title} ---")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"F1-score: {f1_score(y_true, y_pred):.3f}")
    print(classification_report(y_true, y_pred, digits=3))


def plot_confusion(y_true, y_pred, labels=None, title="Matriz de confusión",
                   save_path=None, cmap="Blues"):
    """
    Grafica matriz de confusión con etiquetas automáticas o personalizadas.
    Si 'labels' es None, las genera automáticamente a partir de los valores observados.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Generar etiquetas automáticas si no se especifican
    if labels is None:
        labels = [str(l) for l in sorted(np.unique(np.concatenate([y_true, y_pred])))]
    
    # Graficar heatmap
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                cbar=False)
    plt.title(title)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
