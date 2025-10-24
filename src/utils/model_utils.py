from sklearn.model_selection import GridSearchCV

def tune_logistic_C(X_train, y_train, penalty='l2', solver='liblinear', Cs=None):
    """
    Busca el mejor parámetro C mediante validación cruzada.
    """
    from sklearn.linear_model import LogisticRegression
    
    if Cs is None:
        Cs = [0.001, 0.01, 0.1, 1, 10, 100]
    
    param_grid = {'C': Cs}
    lr = LogisticRegression(penalty=penalty, solver=solver, max_iter=500)
    grid = GridSearchCV(lr, param_grid, scoring='f1', cv=5)
    grid.fit(X_train, y_train)
    
    return grid.best_estimator_, grid.best_params_, grid.best_score_
