import numpy as np
import cvxpy as cvx


def multiple_regress(x, y):
    """
    Function to return the fitted coefficients and intercept for
    a matrix of covariates and a matrix of targets

    Arguments:
    - x: matrix of size (n x d_x)
    - y: matrix of size (n x d_y)
    """
    G = np.einsum("ij,ik->jk", y, x)  # size d_y x d_x
    H_inv = np.einsum("ij,ik->jk", x, x)  # size d_x x d_x
    H = np.linalg.inv(H_inv)  # size d_x x d_x
    mu_x = np.mean(x, axis=0)  # size d_x
    mu_y = np.mean(y, axis=0)  # size d_y
    n = x.shape[0]
    W = (
        (G - n * np.outer(mu_y, mu_x))
        @ H
        @ np.linalg.inv(np.eye(mu_x.shape[0]) - n * np.outer(mu_x, mu_x) @ H)
    )
    b = mu_y - W @ mu_x
    return (W, b)


def multiple_regress_l1(x, y, reg_param):
    """
    Function to return the fitted coefficients and intercept for
    a matrix of covariates and a matrix of targets with complete l1 regularization

    Arguments:
    - x: matrix of size (n x d_x)
    - y: matrix of size (n x d_y)
    - reg_param: float
    """
    n, d_x = x.shape
    n, d_y = y.shape
    W = cvx.Variable((d_y, d_x))
    b = cvx.Variable((1, d_y))
    obj = cvx.sum_squares(y - x @ W.T - np.ones((711, 1)) @ b)
    reg_obj = obj + reg_param * cvx.sum(cvx.abs(W))
    problem = cvx.Problem(cvx.Minimize(reg_obj))
    problem.solve()
    if problem.status == cvx.OPTIMAL:
        return (W.value, b.value)
    else:
        raise Exception("Solver not converged")


def multiple_regress_l2(x, y, reg_param):
    """
    Function to return the fitted coefficients and intercept for
    a matrix of covariates and a matrix of targets with complete l1 regularization

    Arguments:
    - x: matrix of size (n x d_x)
    - y: matrix of size (n x d_y)
    - reg_param: float
    """
    G = np.einsum("ij,ik->jk", y, x)  # size d_y x d_x
    H_inv = np.einsum("ij,ik->jk", x, x) + reg_param * np.eye(x.shape[1])  # size d_x x d_x
    H = np.linalg.inv(H_inv)  # size d_x x d_x
    mu_x = np.mean(x, axis=0)  # size d_x
    mu_y = np.mean(y, axis=0)  # size d_y
    n = x.shape[0]
    W = (
        (G - n * np.outer(mu_y, mu_x))
        @ H
        @ np.linalg.inv(np.eye(mu_x.shape[0]) - n * np.outer(mu_x, mu_x) @ H)
    )
    b = mu_y - W @ mu_x
    return (W, b)


def r2_score(x, y, w, b):
    """
    Function to return the average r2-score across output variables given
    a matrix of covariates, a matrix of targets and a coefficient-intercept pair

    Arguments:
    - x: matrix of size (n x d_x)
    - y: matrix of size (n x d_y)
    - w: matrix of size (d_y x d_x)
    - b: vector of size (d_y)
    """
    hat_y = x @ w.T + b
    return r2_score_with_preds(y, hat_y)


def r2_score_with_preds(y, hat_y):
    """
    Function to return the average r2-score across output variables given
    true outputs and predicted outputs

    Arguments:
    - y: matrix of size (n x d_y)
    - hat_y: matrix of size (n x d_y)
    """
    mean_y = np.mean(y, axis=0)
    SS_tot = np.sum(np.square(y - mean_y))
    SS_res = np.sum(np.square(y - hat_y))
    return np.mean(1 - SS_res / SS_tot)


def cosine_similarity_with_preds(y, hat_y):
    """
    Function to return the average cosine-similarity across output variables
    given true outputs and predicted outputs

    Arguments:
    - y: matrix of size (n x d_y)
    - hat_y: matrix of size (n x d_y)
    """
    inner_prod = np.sum(y * hat_y)
    norm_y = np.sqrt(np.sum(np.square(y)))
    norm_hat_y = np.sqrt(np.sum(np.square(hat_y)))
    return inner_prod / (norm_y * norm_hat_y)


def skill_with_preds(y, hat_y):
    """
    Function to return the average climate skill across output variables
    given true outputs and predicted outputs

    Arguments:
    - y: matrix of size (n x d_y)
    - hat_y: matrix of size (n x d_y)
    """
    return 1 - np.sum(np.square(y - hat_y)) / np.sum(np.square(y))
