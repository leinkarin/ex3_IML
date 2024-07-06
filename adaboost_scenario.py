import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from adaboost import AdaBoost
from decision_stump import DecisionStump

def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners)
    model.fit(train_X, train_y)

    train_errors=[]
    test_errors=[]
    for t in range(1, 251):
        train_errors.append(model.partial_loss(train_X, train_y, t))
        test_errors.append(model.partial_loss(test_X, test_y, t))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, n_learners + 1), train_errors, label='Train Error')
    plt.plot(range(1, n_learners + 1), test_errors, label='Test Error')
    plt.xlabel('Number of Learners')
    plt.ylabel('Error')
    plt.title('Training and Test Errors as a Function of the Number of Learners')
    plt.legend()
    plt.show()


    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, t in enumerate(T):
        # Create mesh grid
        x0, x1 = np.meshgrid(
            np.linspace(lims[0, 0], lims[0, 1], 500),
            np.linspace(lims[1, 0], lims[1, 1], 500)
        )
        grid = np.c_[x0.ravel(), x1.ravel()]

        # Predict using the ensemble up to t iterations
        preds = model.partial_predict(grid, t)
        preds = preds.reshape(x0.shape)

        # Plot decision boundary
        axes[i].contourf(x0, x1, preds, alpha=0.3, levels=np.linspace(-1, 1, 3), cmap=plt.cm.coolwarm)
        axes[i].scatter(test_X[:, 0], test_X[:, 1], c=test_y, cmap=plt.cm.coolwarm, s=20, edgecolor='k')
        axes[i].set_title(f"Decision boundary after {t} learners")
        axes[i].set_xlim(lims[0])
        axes[i].set_ylim(lims[1])

    plt.tight_layout()
    plt.show()


    # Question 3: Decision surface of best performing ensemble
    best_t= np.argmin(test_errors)+1
    accuracy= 1-test_errors[best_t-1]

    x0, x1 = np.meshgrid(
        np.linspace(lims[0, 0], lims[0, 1], 500),
        np.linspace(lims[1, 0], lims[1, 1], 500)
    )
    grid = np.c_[x0.ravel(), x1.ravel()]

    preds = model.partial_predict(grid, best_t)
    preds = preds.reshape(x0.shape)

    plt.figure(figsize=(6, 6))
    plt.contourf(x0, x1, preds, alpha=0.3, levels=np.linspace(-1, 1, 3), cmap=plt.cm.coolwarm)
    plt.scatter(test_X[:, 0], test_X[:, 1], c=test_y, cmap=plt.cm.coolwarm, s=20, edgecolor='k')
    plt.title(f"Decision Boundary for Ensemble Size {best_t}\nAccuracy: {accuracy:.2f}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(lims[0])
    plt.ylim(lims[1])
    plt.show()

    # Question 4: Decision surface with weighted samples
    model = AdaBoost(DecisionStump, 250)
    model.fit(train_X, train_y)

    normalized_weights = model.D_ / np.max(model.D_) * 5
    plt.figure(figsize=(6, 6))
    plt.contourf(x0, x1, preds, alpha=0.3, levels=np.linspace(-1, 1, 3), cmap=plt.cm.coolwarm)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap=plt.cm.coolwarm, s=normalized_weights*20, edgecolor='k')
    plt.title(f"Ensemble Samples Distribution")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.xlim(lims[0])
    plt.ylim(lims[1])
    plt.show()




if __name__ == '__main__':
    np.random.seed(0)
    for noise in [0, 0.4]:
        fit_and_evaluate_adaboost(noise)