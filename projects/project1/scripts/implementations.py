
import numpy as np

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x

def compute_loss_MSE(y, tx, w):
    """Calculate the loss"""
    e = (y - tx.dot(w))
    return e.dot(e) / len(y)

def compute_loss_MAE(y, tx, w):
    """Calculate the loss"""
    e = np.abs(y - tx.dot(w))
    return e.sum() / len(y)

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    return tx.T.dot(y - tx.dot(w)) / -len(y)


def gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss = compute_loss_MSE):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * gradient
        
        # store w and loss
        ws.append(w)
        losses.append(loss)

        #print("Gradient Descent({bi}/{ti}): loss={l}".format(bi=n_iter, ti=max_iters - 1, l=loss))

    return losses, ws



def least_squares(y, tx):
    """calculate the least squares solution."""
    # returns mse, and optimal weights
    opt = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y);
    opt = np.linalg.lstsq(tx, y)[0]
    return compute_loss_MSE(y, tx, opt), opt


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function returns the matrix formed by applying the polynomial basis to the input data
    pol_basis = np.ones((len(x), degree + 1))

    for i in range(1, degree + 1):
        pol_basis[:,i] = pol_basis[:, i - 1] * x

    return pol_basis



def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    txtxt = tx.T.dot(tx)
    lambda_identity = (2 * len(y) * lambda_) * np.identity(len(txtxt))
    first = txtxt + lambda_identity
    firstinv = np.linalg.inv(first)
    firstinvdottxt = firstinv.dot(tx.T)
    return firstinvdottxt.dot(y)