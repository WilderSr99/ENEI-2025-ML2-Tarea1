"""
model_utils.py — Funciones auxiliares de modelado y búsqueda de hiperparámetros.
Incluye utilidad para ajuste de C en Regresión Logística.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd


def tune_logistic_C(X_train, y_train, penalty='l2', solver='liblinear',
                    Cs=None, cv=5, random_state=42, n_jobs=-1, return_df=False):
    """
    Busca el mejor parámetro C mediante validación cruzada optimizando F1.
    Devuelve el mejor modelo, sus parámetros y (opcionalmente) el resumen de CV.

    Parámetros:
    -----------
    X_train, y_train : array-like
        Datos de entrenamiento
    penalty : {'l1', 'l2'}
        Tipo de regularización
    solver : str
        Solver compatible con el tipo de penalty (e.g., 'liblinear')
    Cs : list, optional
        Lista de valores de C a evaluar
    cv : int
        Número de folds para cross-validation
    random_state : int
        Semilla para reproducibilidad
    n_jobs : int
        Núcleos de CPU para paralelizar la búsqueda
    return_df : bool
        Si True, devuelve también DataFrame con los resultados de CV
    """
    if Cs is None:
        Cs = [0.001, 0.01, 0.1, 1, 10, 100]

    param_grid = {'C': Cs}
    lr = LogisticRegression(
        penalty=penalty,
        solver=solver,
        max_iter=500,
        random_state=random_state
    )

    grid = GridSearchCV(
        lr,
        param_grid,
        scoring='f1',
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0
    )

    grid.fit(X_train, y_train)

    if return_df:
        results = pd.DataFrame(grid.cv_results_)
        return grid.best_estimator_, grid.best_params_, grid.best_score_, results
    else:
        return grid.best_estimator_, grid.best_params_, grid.best_score_
