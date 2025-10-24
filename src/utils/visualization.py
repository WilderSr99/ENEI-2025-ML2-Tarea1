"""
visualization.py — Funciones de visualización de fronteras de decisión y regiones de clasificación.
Ideal para Parte B (SVM & overfitting).
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, title="", h=0.02, padding=1.0,
                           cmap=plt.cm.coolwarm, alpha=0.3, save_path=None):
    """
    Grafica la frontera de decisión de un modelo 2D.
    Parámetros:
    -----------
    model : clasificador entrenado (con método .predict)
    X : ndarray shape (n_samples, 2)
    y : etiquetas verdaderas
    h : tamaño del paso de malla
    padding : margen alrededor del rango de datos
    cmap : colormap de matplotlib
    alpha : transparencia de la región
    save_path : ruta para guardar imagen (opcional)
    """
    x_min, x_max = X[:, 0].min() - padding, X[:, 0].max() + padding
    y_min, y_max = X[:, 1].min() - padding, X[:, 1].max() + padding
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=alpha, cmap=cmap)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=25, cmap=cmap)
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
