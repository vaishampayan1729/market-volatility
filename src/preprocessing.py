
"""
Preprocessing utilities for machine-learning experiments.
"""

from sklearn.preprocessing import StandardScaler

#===============================================================================#
#                            standardize_features.                              #
#===============================================================================#

def standardize_features(X_train, X_other=None):
    """
    Standardize feature data using statistics fitted on the training data.

    Parameters
    ----------
    X_train : array-like
        Training features.

    X_other : array-like, optional
        Additional feature data to transform using the scaler fitted on
        X_train. This may be validation or test data.

    Returns
    -------
    scaler : StandardScaler
        Fitted standardization object.

    X_train_scaled : array-like
        Standardized training features.

    X_other_scaled : array-like or None
        Standardized additional features, if provided.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    if X_other is not None:
        X_other_scaled = scaler.transform(X_other)
    else:
        X_other_scaled = None

    return scaler, X_train_scaled, X_other_scaled


