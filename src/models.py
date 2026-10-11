"""
Model construction utilities for machine-learning experiments.
"""
# tree models
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
# neural networks
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import AdamW
#===============================================================================#
#                            create_random_forest.                              #
#===============================================================================#

def create_random_forest(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight=None,
    random_state=42,
    n_jobs=-1,
):
    """
    Create a Random Forest classifier.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of trees in the forest.

    max_depth : int or None, default=None
        Maximum depth of each tree. None allows trees to grow until
        other stopping criteria are reached.

    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.

    min_samples_leaf : int, default=1
        Minimum number of samples required at a leaf node.

    max_features : int, float, str, or None, default="sqrt"
        Number of features considered when looking for the best split.

    class_weight : dict, "balanced", "balanced_subsample", or None
        Weights associated with the classes.

    random_state : int, default=42
        Controls the randomness of the forest.

    n_jobs : int, default=-1
        Number of jobs used for fitting and prediction.
        -1 uses all available processors.

    Returns
    -------
    RandomForestClassifier
        An unfitted Random Forest classifier.
    """

    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )


#===============================================================================#
#                               create xgboost                                  #
#===============================================================================#

def create_xgboost(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    reg_lambda=1.0,
    reg_alpha=0.0,
    random_state=42,
    n_jobs=-1,
):
    """
    Create an XGBoost classifier.

    Parameters
    ----------
    n_estimators : int, default=200
        Number of boosting rounds.

    max_depth : int, default=3
        Maximum depth of each decision tree.

    learning_rate : float, default=0.05
        Step size used during boosting.

    subsample : float, default=0.8
        Fraction of training observations sampled for each boosting round.

    colsample_bytree : float, default=0.8
        Fraction of features sampled for each tree.

    min_child_weight : float, default=1
        Minimum sum of instance weights required in a child node.

    reg_lambda : float, default=1.0
        L2 regularization term on leaf weights.

    reg_alpha : float, default=0.0
        L1 regularization term on leaf weights.

    random_state : int, default=42
        Controls randomization.

    n_jobs : int, default=-1
        Number of parallel threads.

    Returns
    -------
    XGBClassifier
        An unfitted XGBoost classifier.
    """

    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss",
    )

#===============================================================================#
#                               create neural network                           #
#===============================================================================#
# build the model

def build_nn_classifier(
    n_features,
    hidden_layers=(32, 16),
    activation="relu",
    learning_rate=0.001,
    weight_decay=0.0,
):
    """
    Build and compile a feedforward neural network
    for binary classification.
    """

    model = Sequential(
        name="stage0_nn_classifier"
    )

    model.add(
        Input(shape=(n_features,))
    )

    for units in hidden_layers:
        model.add(
            Dense(
                units=units,
                activation=activation,
            )
        )

    model.add(
        Dense(
            1,
            activation="sigmoid",
        )
    )

    model.compile(
        optimizer=AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model

# training utilities

from tensorflow.keras.callbacks import EarlyStopping


def train_nn_classifier(
    model,
    X_train,
    y_train,
    X_validation,
    y_validation,
    epochs=200,
    batch_size=256,
    patience=20,
    verbose=1,
):
    """
    Train a neural-network classifier with early stopping.
    """

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_validation, y_validation),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        shuffle=False,
        verbose=verbose,
    )

    return history