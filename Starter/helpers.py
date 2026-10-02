import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score



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


def custom_make_pipeline(preprocessor: ColumnTransformer, feature_pipeline, classifier):
    return Pipeline([
        ("preprocess", preprocessor),
        ("features", feature_pipeline),
        ("model", classifier),
    ])