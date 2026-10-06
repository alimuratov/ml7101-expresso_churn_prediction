from pathlib import Path

import joblib
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


# Result variables used by the notebook, including older checkpoints.
RESULT_NAMES = """
pipelines score_rows feature_counts comparison feature_choices
l1_rows l1_summaries l1_feature_counts l1_comparison frequency
l1_fixed_summary l1_fixed_counts
tree_summary tree_counts tree_comparison tree_frequency
rfe_feature_comparison all_selection_frequency
results cv_splits
""".split()


def save_results(path, namespace):
    """Save available experiment results from the notebook's globals()."""
    state = {name: namespace[name] for name in RESULT_NAMES if name in namespace}
    if not state:
        print("No experiment results to save yet.")
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(state, temporary, compress=3)
    temporary.replace(path)
    print("Saved:", ", ".join(state))


def load_results(path):
    """Read a local checkpoint, or return an empty dictionary on the first run."""
    path = Path(path)
    if not path.exists():
        print("No checkpoint found. Run the experiments and call save_results().")
        return {}
    state = joblib.load(path)
    print("Restored:", ", ".join(state))
    return state


def make_selection_pipelines(selectors, preprocessor, feature_engineering, classifier):
    """Change only feature selection, keeping the other pipeline steps identical."""
    return {
        name: custom_make_pipeline(
            preprocessor=clone(preprocessor),
            feature_engineering=clone(feature_engineering),
            feature_pipeline=clone(selector) if selector != "passthrough" else selector,
            classifier=clone(classifier),
        )
        for name, selector in selectors.items()
    }


def evaluate_experiments(pipelines, X, y, splits, *, index_name="method"):
    """Evaluate named pipelines on shared splits and collect scores and selections."""
    summaries, feature_counts = {}, {}
    for name, pipeline in pipelines.items():
        summary, counts = evaluate_pipeline(pipeline, X, y, splits)
        summaries[name], feature_counts[name] = summary, counts
        print(
            f"{name}: AUC={summary['mean_auc']:.5f}, "
            f"columns={summary['mean_columns_kept']:.1f}"
        )
    comparison = pd.DataFrame.from_dict(summaries, orient="index")
    comparison.index.name = index_name
    return comparison, feature_counts


def evaluate_if_missing(name, pipeline, X, y, splits, namespace):
    """Reuse name_summary and name_counts when data and settings are unchanged.

    Remove those variables before rerunning an experiment with changed inputs.
    """
    summary_key, counts_key = f"{name}_summary", f"{name}_counts"
    if summary_key in namespace and counts_key in namespace:
        return namespace[summary_key], namespace[counts_key]
    return evaluate_pipeline(pipeline, X, y, splits)


def count_selected_features(selection_pipeline, X, y, splits):
    """Recover selection counts without training the final prediction model."""
    counts = Counter()
    for train_indices, _ in splits:
        fitted = clone(selection_pipeline).fit(X.iloc[train_indices], y.iloc[train_indices])
        names = fitted.named_steps["preprocess"].get_feature_names_out()
        counts.update(names[fitted.named_steps["features"].get_support()])
    return counts


def selection_frequencies(counts, *, features=None, sort=True):
    """Make a feature-by-method count table, filling unselected features with zero."""
    frequency = pd.DataFrame(counts).fillna(0)
    if features is not None:
        frequency = frequency.reindex(features, fill_value=0)
    frequency = frequency.astype(int)
    if sort:
        order = frequency.min(axis=1).sort_values(ascending=False, kind="stable").index
        frequency = frequency.loc[order]
    return frequency


def plot_selection_agreement(frequency, n_folds, *, title="Agreement between selection methods"):
    """Show selection counts and highlight features retained by every method in every fold."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    agreed = frequency.eq(n_folds).all(axis=1)
    print(f"Selected by all {frequency.shape[1]} methods in every fold:")
    print(frequency.index[agreed].tolist())
    plt.figure(figsize=(max(8, 2.5 * frequency.shape[1]), max(4, 0.4 * len(frequency))))
    ax = sns.heatmap(
        frequency, annot=True, fmt="d", cmap="Blues", vmin=0, vmax=n_folds,
        cbar_kws={"label": "Number of folds selected"},
    )
    for label, selected in zip(ax.get_yticklabels(), agreed):
        if selected:
            label.set_fontweight("bold")
    ax.set(
        title=f"{title}\nBold rows: selected by every method in every fold",
        xlabel="Selection method", ylabel="",
    )
    plt.tight_layout()
    plt.show()


def plot_auc_changes(comparison, *, baseline="All columns"):
    """Compare AUC against the baseline within each shared validation fold."""
    import matplotlib.pyplot as plt

    fold_auc = pd.DataFrame(comparison["fold_scores"].to_dict())
    fold_auc.index = range(1, len(fold_auc) + 1)
    changes = fold_auc.drop(columns=baseline).sub(fold_auc[baseline], axis=0)
    ax = changes.plot(marker="o", figsize=(9, 4))
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set(
        title=f"Boosting AUC change vs {baseline.lower()}",
        xlabel="Validation fold", ylabel="AUC change (positive = improvement)",
        xticks=fold_auc.index,
    )
    plt.tight_layout()
    plt.show()


def show_rfe_comparison(logistic_counts, tree_counts, n_folds, *, features=None):
    """Compare aggregate selection counts, which do not identify individual folds."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    comparison = selection_frequencies({
        "Logistic regression": logistic_counts,
        "Decision tree": tree_counts,
    }, features=features, sort=False)
    comparison["Tree minus logistic"] = (
        comparison["Decision tree"] - comparison["Logistic regression"]
    )
    comparison = comparison.sort_values("Tree minus logistic", ascending=False, kind="stable")
    print(f"Number of folds selected, out of {n_folds}:")
    display(comparison)
    both_always = comparison[["Logistic regression", "Decision tree"]].eq(n_folds).all(axis=1)
    print("Selected by both RFE methods in every fold:")
    print(comparison.index[both_always].tolist())

    differences = comparison["Tree minus logistic"]
    differences = differences[differences != 0].sort_values()
    if differences.empty:
        print("Both methods selected each feature equally often.")
    else:
        ax = differences.plot.barh(
            figsize=(9, max(3, 0.4 * len(differences))),
            color=["#2563eb" if value > 0 else "#c2410c" for value in differences],
        )
        ax.axvline(0, color="black", linewidth=1)
        ax.set(
            title="RFE: differences in feature selection frequency",
            xlabel="Difference in folds selected (decision tree minus logistic regression)",
            ylabel="", xlim=(-n_folds - 0.5, n_folds + 0.5),
            xticks=range(-n_folds, n_folds + 1),
        )
        plt.tight_layout()
        plt.show()
    return comparison
