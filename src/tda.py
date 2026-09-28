import numpy as np
from ripser import ripser

#=====================================================================#
# delay_embedding function
#=====================================================================#

def delay_embedding(
    series,
    tau=1,
    dimension=2
):
    """
    Construct a delay-coordinate embedding of a time series.

    Parameters
    ----------
    series : array-like
        One-dimensional time series.
    tau : int, default=1
        Delay between consecutive coordinates.
    dimension : int, default=2
        Embedding dimension.

    Returns
    -------
    np.ndarray
        Array of shape
        (n_points, dimension), where each row is

            (r_t, r_{t-tau}, ..., r_{t-(dimension-1)tau}).

    Raises
    ------
    ValueError
        If tau or dimension is not positive, or if the series
        is too short for the requested embedding.
    """

    series = np.asarray(series)

    if series.ndim != 1:
        raise ValueError("series must be one-dimensional.")

    if tau < 1:
        raise ValueError("tau must be a positive integer.")

    if dimension < 1:
        raise ValueError("dimension must be a positive integer.")

    n_points = len(series) - (dimension - 1) * tau

    if n_points <= 0:
        raise ValueError(
            "The series is too short for the requested "
            "tau and embedding dimension."
        )

    embedding = np.column_stack(
        [
            series[
                (dimension - 1 - i) * tau:
                (dimension - 1 - i) * tau + n_points
            ]
            for i in range(dimension)
        ]
    )

    return embedding

#=============================================================================#
# sliding window function
#=============================================================================#

def sliding_windows(series, window_size, step=1):
    """
    Divide a one-dimensional time series into overlapping
    sliding windows.

    Parameters
    ----------
    series : array-like
        One-dimensional time series.
    window_size : int
        Number of observations in each window.
    step : int, default=1
        Number of observations by which the window advances.

    Returns
    -------
    list of np.ndarray
        List containing the sliding windows.

    Raises
    ------
    ValueError
        If the series is not one-dimensional, or if window_size
        or step is not positive, or if the series is shorter
        than the requested window.
    """

    series = np.asarray(series)

    if series.ndim != 1:
        raise ValueError("series must be one-dimensional.")

    if window_size < 1:
        raise ValueError("window_size must be positive.")

    if step < 1:
        raise ValueError("step must be positive.")

    if len(series) < window_size:
        raise ValueError(
            "The series is shorter than the requested window_size."
        )

    windows = [
        series[start:start + window_size]
        for start in range(
            0,
            len(series) - window_size + 1,
            step
        )
    ]

    return windows

#===========================================================================#
# sliding_window_dates function
#===========================================================================#
def sliding_window_dates(index, window_size, step=1):
    """
    Return the ending date associated with each sliding window.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Datetime index of the time series.
    window_size : int
        Number of observations in each window.
    step : int, default=1
        Number of observations by which the window advances.

    Returns
    -------
    pandas.DatetimeIndex
        Ending date of each sliding window.
    """

    if window_size < 1:
        raise ValueError("window_size must be positive.")

    if step < 1:
        raise ValueError("step must be positive.")

    if len(index) < window_size:
        raise ValueError(
            "The index is shorter than the requested window_size."
        )

    return index[
        window_size - 1::step
    ]

#=================================================================================#
# persistence statistics function
#=================================================================================#
def persistence_statistics(diagram):
    """
    Compute summary statistics for a persistence diagram.

    Infinite-persistence features are excluded from the numerical
    summaries.

    Parameters
    ----------
    diagram : np.ndarray
        Persistence diagram of shape (n_features, 2), where each row
        contains a birth and death time.

    Returns
    -------
    dict
        Dictionary containing:
        - n_features: total number of persistence pairs
        - n_finite: number of finite persistence pairs
        - n_infinite: number of infinite persistence pairs
        - total_persistence: sum of finite persistence values
        - mean_persistence: mean finite persistence
        - max_persistence: largest finite persistence
    """

    diagram = np.asarray(diagram)

    if diagram.ndim != 2 or diagram.shape[1] != 2:
        raise ValueError(
            "diagram must have shape (n_features, 2)."
        )

    persistence = diagram[:, 1] - diagram[:, 0]

    finite_mask = np.isfinite(persistence)
    finite_persistence = persistence[finite_mask]

    n_features = len(persistence)
    n_finite = len(finite_persistence)
    n_infinite = n_features - n_finite

    if n_finite == 0:
        total_persistence = 0.0
        mean_persistence = 0.0
        max_persistence = 0.0
    else:
        total_persistence = finite_persistence.sum()
        mean_persistence = finite_persistence.mean()
        max_persistence = finite_persistence.max()

    return {
        "n_features": n_features,
        "n_finite": n_finite,
        "n_infinite": n_infinite,
        "total_persistence": total_persistence,
        "mean_persistence": mean_persistence,
        "max_persistence": max_persistence,
    }

#=======================================================================#
# compute persistence diagram
#=======================================================================#
def compute_persistence_diagram(
    point_cloud,
    max_dimension=1
):
    """
    Compute persistent homology of a point cloud.

    Parameters
    ----------
    point_cloud : array-like
        Point cloud of shape (n_points, n_dimensions).
    max_dimension : int, default=1
        Maximum homology dimension to compute.

    Returns
    -------
    list of np.ndarray
        Persistence diagrams indexed by homology dimension.
    """

    point_cloud = np.asarray(point_cloud)

    if point_cloud.ndim != 2:
        raise ValueError(
            "point_cloud must be a two-dimensional array."
        )

    if max_dimension < 0:
        raise ValueError(
            "max_dimension must be non-negative."
        )

    result = ripser(
        point_cloud,
        maxdim=max_dimension
    )

    return result["dgms"]