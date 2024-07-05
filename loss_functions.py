import numpy as np


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    # Calculate the number of misclassified samples
    misclassified = np.sum(y_true != y_pred)

    if normalize:
        # Normalize by the number of samples
        return misclassified / len(y_true)
    else:
        # Return the count of misclassified samples
        return misclassified




