from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from feature_engine.encoding import RareLabelEncoder, WoEEncoder


def get_pipeline(
    numeric_features,
    categorical_features,
    use_woe: bool = True,
):
    numeric_features = list(numeric_features)
    categorical_features = list(categorical_features)

    num_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    if use_woe:
        cat_pipeline = Pipeline(
            steps=[
                ("rare", RareLabelEncoder(tol=0.05, n_categories=1)),
                ("woe", WoEEncoder(unseen="ignore", fill_value=0.0001)),
            ]
        )
    else:
        cat_pipeline = Pipeline(
            steps=[
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preprocessor


def create_pipeline(
    numeric_features,
    categorical_features,
    use_woe: bool = True,
):
    # Backward-compatible alias for existing scripts.
    return get_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        use_woe=use_woe,
    )
