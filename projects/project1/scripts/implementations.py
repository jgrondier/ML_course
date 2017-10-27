
import numpy as np

from time import time

try: from tqdm import tqdm
except: tqdm = lambda x: x



def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    if std_x == 0:
        return np.zeros(len(x))
    x = x / std_x
    return x

def compute_loss_MSE(y, tx, w):
    """Calculate the loss"""
    e = (y - tx.dot(w))
    return e.T.dot(e) / len(y)

def compute_loss_MAE(y, tx, w):
    """Calculate the loss"""
    e = np.abs(y - tx.dot(w))
    return e.sum() / len(y)

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    loss = 0    
    for i in range(0 , len(y)):
        loss += (np.log(1 + np.exp(np.dot(tx[i],w))) - y[i]*(np.dot(tx[i],w)))
        
    return loss

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    return tx.T.dot(y - tx.dot(w)) / -len(y)

def gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss = compute_loss_MSE):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in tqdm(range(max_iters)):
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
    # opt = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y);
    opt = np.linalg.inv(np.dot(tx.T, tx))
    opt = np.dot(opt, tx.T)
    opt = np.dot(opt, y)

    return compute_loss_MSE(y, tx, opt), opt

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    
    return gradiant_descent(y, tx, initial_w, max_iters, gamma, compute_loss = calculate_loss)
    
    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    
    batch_size = 10
    new_y, new_tx = y.reshape(-1, batch_size), tx.reshape(-1, batch_size)
    combined = list(zip(y, tx))
    random.shuffle(combined)
    new_y[:], new_tx[:] = zip(*combined)
    return gradiant_descent(y, tx, initial_w, max_iters, gamma, compute_loss = calculate_loss)


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

def sigmoid(t):
    """apply sigmoid function on t."""
    
    ex = np.exp(t)
    
    return ex / (ex + 1)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""

    return tx.T.dot(sigmoid(np.dot(tx,w)) - y)

def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    return calculate_loss(y, tx, w), calculate_gradient(y, tx, w), calculate_hessian(y, tx, w)

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    S = np.identity(len(y))
    
    for i in range(len(y)):
        xtw = tx[i].T.dot(w)
        S[i, i] = sigmoid(xtw) * ( 1 - sigmoid(xtw))
    
    ret = tx.T.dot(S)
    return ret.dot(tx)


def logistic_regression(y, tx, w):
    """return the loss, gradient, and hessian."""
    return calculate_loss(y, tx, w), calculate_gradient(y, tx, w), calculate_hessian(y, tx, w)

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(int(seed))
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


    
def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient, and hessian: TODO
    # ***************************************************
    
    loss = calculate_loss(y, tx, w)
    
    loss += 0.5 * lambda_ * np.linalg.norm(w) ** 2
    
    g = calculate_gradient(y, tx, w)
    
    g += lambda_ * w
    
    hessian = calculate_hessian(y, tx, w)
    
    hessian +=  0.5 * lambda_ * np.linalg.norm(w) ** 2
    
    return loss, g, hessian

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # return loss, gradient: TODO
    # ***************************************************
    
    loss, g, hessian = penalized_logistic_regression(y, tx, w, lambda_)
    
    # ***************************************************
    # INSERT YOUR CODE HERE
    # update w: TODO
    # ***************************************************
    
    w = w - gamma*g
    
    
    return loss, w
    
def cross_validation_datasets(y, tx, k_fold, seed = time()):
    k_indices = build_k_indices(y, k_fold, seed)
    """return the loss of ridge regression."""
    for k in range(k_fold):
        # ***************************************************
        # get k'th subgroup in test, others in train
        # ***************************************************
        test_y = y[k_indices[k]]
        test_x = tx[k_indices[k]]
        train_y = (y[k_indices[np.arange(len(k_indices))!=k]]).flatten()
        train_x = (tx[k_indices[np.arange(len(k_indices))!=k]]).reshape(len(train_y), tx.shape[1])
        yield test_y, test_x, train_y, train_x