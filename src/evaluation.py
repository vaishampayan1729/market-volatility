
import pandas as pd
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)

#===============================================================================#
#                            Classification Metrics                             #
#===============================================================================#

def classification_metrics(y_true, y_pred, y_proba=None):
    """
    Compute binary classification metrics.

    Parameters
    ----------
    y_true : array-like
        True binary labels.

    y_pred : array-like
        Predicted binary labels.

    y_proba : array-like, optional
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Classification metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_proba,
        )

        metrics["pr_auc"] = average_precision_score(
            y_true,
            y_proba,
        )

        metrics["log_loss"] = log_loss(
            y_true,
            y_proba,
        )

    return metrics


#===============================================================================#
#                      evaluate classification model.                           #
#===============================================================================#

def evaluate_classification_model(model, X, y, cv):
    """
    Evaluate a classification model using cross-validation.

    Parameters
    ----------
    model : estimator
        Scikit-learn compatible classification model.

    X : array-like
        Feature data.

    y : array-like
        Target labels.

    cv : cross-validation splitter
        Cross-validation strategy providing train/validation splits.

    Returns
    -------
    pandas.DataFrame
        Metrics for each validation fold.
    """

    fold_results = []

    for fold, (train_indices, validation_indices) in enumerate(
        cv.split(X),
        start=1,
    ):
        X_train = X.iloc[train_indices]
        X_validation = X.iloc[validation_indices]

        y_train = y.iloc[train_indices]
        y_validation = y.iloc[validation_indices]

        fold_model = clone(model)

        fold_model.fit(X_train, y_train)

        y_validation_pred = fold_model.predict(X_validation)

        if hasattr(fold_model, "predict_proba"):
            y_validation_proba = fold_model.predict_proba(
                X_validation
            )[:, 1]
        else:
            y_validation_proba = None

        metrics = classification_metrics(
            y_validation,
            y_validation_pred,
            y_validation_proba,
        )

        metrics["fold"] = fold
        fold_results.append(metrics)

    return pd.DataFrame(fold_results)


#===============================================================================#
#                   Add evaluate_classification_test_set()                      #
#===============================================================================#

def evaluate_classification_test_set(model, X_test, y_test):
    """
    Evaluate a fitted classification model on a test set.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn compatible classification model.

    X_test : array-like
        Test features.

    y_test : array-like
        True test labels.

    Returns
    -------
    dict
        Classification metrics on the test set.
    """

    y_test_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_test_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_test_proba = None

    return classification_metrics(
        y_test,
        y_test_pred,
        y_test_proba,
    )




