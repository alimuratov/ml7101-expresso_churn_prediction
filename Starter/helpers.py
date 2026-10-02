import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import FunctionTransformer
from collections import Counter




def make_feature_pipeline(selection="all", *, k=0.5, estimator=None, step=0.2):
    """
    k is the number of columns to keep after preprocessing
    """
    if selection == "all":
        return "passthrough"
    if selection != "rfe":
        raise ValueError("selection must be 'all' or 'rfe'.")

    selector_model = (
        LogisticRegression(max_iter=2000)
        if estimator is None else clone(estimator)
    )
    return Pipeline([
        ("select", RFE(selector_model, n_features_to_select=k, step=step)),
    ])


def custom_make_pipeline(preprocessor: ColumnTransformer, feature_pipeline, classifier, *, feature_engineering="passthrough"):
    return Pipeline([
        ("engineer", feature_engineering),
        ("preprocess", preprocessor),
        ("features", feature_pipeline),
        ("model", classifier),
    ])

def add_features(X):
    """
    to use: 
    feature_engineering = Pipeline([
        ("add_columns", FunctionTransformer(add_features)),
    ])
    """
    result = X.copy()

    zero_columns = [
        "ARPU_SEGMENT", "DATA_VOLUME", "ON_NET",
        "ORANGE", "TIGO", "ZONE1", "ZONE2",
    ]

    for column in zero_columns:
        result[f"{column}_IS_MISSING"] = X[column].isna().astype(int)

    # average topup
    refill_count = X["FREQUENCE_RECH"].where(X["FREQUENCE_RECH"] > 0)
    result["AVERAGE_TOP_UP_AMOUNT"] = X["MONTANT"] / refill_count

    # inactivity count
    inactive = X.isna() | X.eq(0) | X.eq("None")
    result["INACTIVITY_COUNT"] = inactive.sum(axis=1)

    return result

def evaluate_pipeline(pipeline, X, y, splits):
    # measure pipeline performance using cross-validation 
    # and keep track of which features it keeps in each fold
    result = cross_validate(
        pipeline,
        X,
        y,
        cv=splits,
        scoring="roc_auc",
        return_estimator=True,  # keep fitted models to inspect their features
        n_jobs=1,
        error_score="raise",
    )

    feature_counts = Counter()
    column_counts = []

    for fitted in result["estimator"]:
        names = fitted.named_steps["preprocess"].get_feature_names_out()
        selector = fitted.named_steps["features"]

        if selector != "passthrough":
            names = names[selector.get_support()]

        feature_counts.update(names)
        column_counts.append(len(names))

    scores = result["test_score"]

    summary = {
        "mean_auc": scores.mean(),
        "std_auc": scores.std(ddof=1),
        "mean_columns_kept": np.mean(column_counts),
        "fold_scores": scores.tolist(),
    }

    return summary, feature_counts