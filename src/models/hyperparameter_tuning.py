"""
hypertuning.py
XGBoost hyperparameter tuning using RandomizedSearchCV
"""

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold


def tune_xgboost(pipeline, X_train, y_train):

    param_dist = {
        "classifier__n_estimators": [100, 200, 300, 400],
        "classifier__max_depth": [3, 4, 5, 6, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.6, 0.8, 1.0],
        "classifier__colsample_bytree": [0.6, 0.8, 1.0],
        "classifier__gamma": [0, 0.1, 0.3, 0.5],
        "classifier__min_child_weight": [1, 3, 5]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    search.fit(X_train, y_train)

    return search
