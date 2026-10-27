import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, tol=1e-8):
    
    y = np.asarray(y, dtype=float).reshape(-1)
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(initial_w, dtype=float).reshape(-1).copy()
    for _ in range(max_iters):
        error = tx @ w - y
        gradient = tx.T @ error / y.size
        w_new = w - gamma * gradient

        # Stop if the weights barely change
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break

        w = w_new

    error = y - tx @ w
    loss = 0.5 * np.mean(error ** 2)

    return w, float(loss)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma, tol=1e-8):
    y = np.asarray(y, dtype=float).reshape(-1)
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(initial_w, dtype=float).reshape(-1).copy()

    for _ in range(max_iters):
        n = np.random.randint(y.size)

        error = tx[n] @ w - y[n]
        gradient = tx[n] * error

        w_new = w - gamma * gradient
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new

    error = y - tx @ w
    loss = 0.5 * np.mean(error ** 2)

    return w, float(loss)

def least_squares(y, tx):
    y = np.asarray(y, dtype=float).reshape(-1)
    tx = np.asarray(tx, dtype=float)

    w, _, _, _ = np.linalg.lstsq(tx, y, rcond=None)

    error = y - tx @ w
    loss = 0.5 * np.mean(error ** 2)

    return w, float(loss)



def ridge_regression(y, tx, lambda_):
    y = np.asarray(y, dtype=float).reshape(-1)
    tx = np.asarray(tx, dtype=float)

    n, d = tx.shape

    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    b = tx.T @ y

    w = np.linalg.solve(a, b)

    error = y - tx @ w
    loss = 0.5 * np.mean(error ** 2)

    return w, float(loss)


def logistic_regression(y, tx, initial_w, max_iters, gamma, tol=1e-8):
    y = np.asarray(y, dtype=float).reshape(-1)
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(initial_w, dtype=float).reshape(-1).copy()

    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("Labels must be -1 or 1.")

    for _ in range(max_iters):
        scores = tx @ w

        # Gradient for labels -1 and 1
        margins = y * scores
        gradient = -(tx.T @ (y * np.exp(-np.logaddexp(0.0, margins)))) / y.size

        w_new = w - gamma * gradient

        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break

        w = w_new

    scores = tx @ w
    loss = np.mean(np.logaddexp(0.0, -y * scores))

    return w, float(loss)


def reg_logistic_regression(y, tx, lambda_, initial_w,
                            max_iters, gamma, tol=1e-8):
    y = np.asarray(y, dtype=float).reshape(-1)
    tx = np.asarray(tx, dtype=float)
    w = np.asarray(initial_w, dtype=float).reshape(-1).copy()

    if not np.all(np.isin(y, [-1, 1])):
        raise ValueError("Labels must be -1 or 1.")

    for _ in range(max_iters):
        scores = tx @ w
        margins = y * scores

        gradient = -(tx.T @ (
            y * np.exp(-np.logaddexp(0.0, margins))
        )) / y.size

        gradient += 2 * lambda_ * w

        w_new = w - gamma * gradient

        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break

        w = w_new

    scores = tx @ w
    loss = np.mean(np.logaddexp(0.0, -y * scores))

    return w, float(loss)