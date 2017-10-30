
import numpy as np

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
        txiw = np.dot(tx[i],w)

        if np.max(txiw) > 700:
            loss += txiw
        else:
            loss += np.log(1 + np.exp(np.dot(tx[i],w)))
        loss -= y[i]*(txiw)

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

    """Return a gradient descent of least squares"""

    return gradiant_descent(y, tx, initial_w, max_iters, gamma, compute_loss = calculate_loss)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):

    """Returns a batched stochastic gradient descent of least squares"""

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

    t[ t > 700] = 700

    ex = np.exp(t)

    return ex / (ex + 1)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    txw = np.dot(tx,w)
    txwy = sigmoid(txw) - y
    return tx.T.dot(txwy)

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

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(int(seed))
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)



def penalized_logistic_regression(y, tx, w, lambda_, loss_function = calculate_loss):
    """return the loss, gradient, and hessian."""

    loss = loss_function(y, tx, w)

    loss += 0.5 * lambda_ * np.linalg.norm(w) ** 2

    g = calculate_gradient(y, tx, w)

    g += lambda_ * w

    hessian=0 #Set to 0 because not used and creates memory errors


    return loss, g , hessian

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_, loss_function = calculate_loss):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """

    loss, g, hessian = penalized_logistic_regression(y, tx, w, lambda_, loss_function)

    w = w - gamma*g


    return loss, w

def cross_validation_datasets(y, tx, k_fold, seed = 42):
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

def reg_logistic_regression(y, tx, lambda_=0.01, initial_w=None, max_iters=50, gamma=10, compute_loss = calculate_loss):

    """Return loss, w of a reg logistic regression with at most max_iters iterations"""

    if initial_w is None:
        initial_w = np.zeros((tx.shape[1], 1))
        

    threshold = 1e-8

    losses = []
    w = initial_w

    for n_iter in tqdm(range(max_iters)):
        loss, g, _ = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * g

        losses.append(loss)

        if n_iter > 0 and np.abs(losses[n_iter] - losses[n_iter-1]) < threshold:
            return loss, w

    return losses[-1], w


def uniq_count(t):
    """returns the number of unique values in t """
    vals = set()
    for x in t:
        vals.add(x)
    return len(vals)

def standardize_matrix(data):
    """standardize matrix column by column"""
    return np.array([standardize(c) for c in data.T]).T


def bucket_events(data):
    """bucket events by PRI_jet_num"""
    return [np.where(data[:, 22] == i)[0] for i in range(0, 4)]

def remove_undef(data):
    """replace -999 by the mean"""
    data = np.copy(data)
    mean = data[data != -999].mean()
    data[np.where(data == -999)] = mean
    return data

def columns(data):
    """Returns an matrix without undefined values and specified colums"""
    return np.array([remove_undef(c) for i, c in enumerate(data.T) if i not in [22]])#[14, 15, 17, 18, 22, 24, 25, 27, 28]])

def analyse_data(data):
    """Returns ivertibles columns and sqrt columns"""
    cols = columns(data)
    pos_cols = [i for i, c in enumerate(cols) if c.min() > 0]
    nez_cols = [i for i, c in enumerate(cols) if 0 not in c]
    return (pos_cols, nez_cols)

def prepare_data(data, analysed, degree):
    """transforms mass and add derivated features"""
    cols = columns(data)
    polys = [np.array([np.power(c, deg) for c in cols]).T for deg in range(2, degree + 1)]
    #boolified = np.array([np.where(c < 0, -1, 1) for c in cols])

    matrix = np.column_stack([cols.T, np.log(cols[analysed[0]]).T, np.reciprocal(cols[analysed[1]]).T] + polys)
    polys = cols = pos_cols = nez_cols = None
    return np.c_[np.ones(len(data)),
                 np.where(data[:, 0] < 0, -1, 1), # mass bool
                 standardize_matrix(matrix)]



def linreg(y, tx):
    """performs basic linear regression"""
    assert(len(y) == len(tx))
    assert(len(y) > 0)

    losses, ws = gradient_descent(y, tx, np.zeros(tx.shape[1]), 5000, 0.01)
    min_loss = min(losses)
    return next(w for l, w in zip(losses, ws) if l == min_loss)


def compute_ridge_rmse(yb, raw_data, lambda_, degree):

    """Returns the RMSE for one specific lambda_, degree combination"""

    analysed = analyse_data(raw_data)

    rmse = [[], [], [], []]
    for (test_y, raw_test_x, train_y, raw_train_x) in cross_validation_datasets(yb, raw_data, 4):
        train_x = prepare_data(raw_train_x, analysed, degree)

        pri_train_buckets = bucket_events(raw_train_x)
        pri_w = [ridge_regression(train_y[b], train_x[b], lambda_) for b in pri_train_buckets]

        pri_train_buckets = train_x = train_y = raw_train_x = None

        test_x = prepare_data(raw_test_x, analysed, degree)
        pri_test_buckets = bucket_events(raw_test_x)

        for i, b in enumerate(pri_test_buckets):
            rmse_te = np.sqrt(compute_loss_MSE(test_y[b], test_x[b], pri_w[i]))
            rmse[i].append(rmse_te)

    return [np.sum(r) for r in rmse]

def compute_ridge_fail_rate(yb, raw_data, lambda_, degree):

    """Returns the fail rate for one specific (degree, lambda_) combination"""

    analysed = analyse_data(raw_data)

    errors = [0, 0, 0, 0]
    totals = [0, 0, 0, 0]
    for (test_y, raw_test_x, train_y, raw_train_x) in cross_validation_datasets(yb, raw_data, 4, 42):
        train_x = prepare_data(raw_train_x, analysed, degree)

        pri_train_buckets = bucket_events(raw_train_x)
        pri_w = [ridge_regression(train_y[b], train_x[b], lambda_) for b in pri_train_buckets]

        pri_train_buckets = train_x = train_y = raw_train_x = None

        test_x = prepare_data(raw_test_x, analysed, degree)
        pri_test_buckets = bucket_events(raw_test_x)


        for i, ev in enumerate(test_x):
            pri = int(raw_test_x[i, 22])
            assert(pri >= 0 and pri < 4)
            x = pri_w[pri].dot(ev)
            pred = -1 if x < 0 else 1
            errors[pri] += 0 if pred == test_y[i] else 1
            totals[pri] += 1

    return [e / t for e, t in zip(errors, totals)]

def compute_logit_fail_rate(yb, raw_data, lambda_, degree, gamma, threshold):

    """Returns the fail rate for one specific (degree, lambda_, gamma) combination with a logistic regression"""

    analysed = analyse_data(raw_data)

    errors = [0, 0, 0, 0]
    totals = [0, 0, 0, 0]
    for (test_y, raw_test_x, train_y, raw_train_x) in cross_validation_datasets(yb, raw_data, 4, 42):
        train_x = prepare_data(raw_train_x, analysed, degree)

        pri_train_buckets = bucket_events(raw_train_x)
        pri_w = []
        for b in tqdm(pri_train_buckets):
            w, loss = train_logistic(train_y[b], train_x[b], gamma, lambda_, 1000, threshold)
            pri_w.append(w)

        pri_train_buckets = train_x = train_y = raw_train_x = None

        test_x = prepare_data(raw_test_x, analysed, degree)
        pri_test_buckets = bucket_events(raw_test_x)


        for i, ev in enumerate(test_x):
            pri = int(raw_test_x[i, 22])
            assert(pri >= 0 and pri < 4)
            x = pri_w[pri].dot(ev)
            pred = 0 if x < 0.5 else 1
            errors[pri] += 0 if pred == test_y[i] else 1
            totals[pri] += 1

    return [e / t for e, t in zip(errors, totals)]

def train_ridge_rmse(yb, raw_data, lambda_, degree):

    """Prepare the data then trains using ridge regression on the given dataset"""

    analysed = analyse_data(raw_data)

    train_x = prepare_data(raw_data, analysed, degree)

    pri_train_buckets = bucket_events(raw_data)
    pri_w = [ridge_regression(yb[b], train_x[b], lambda_) for b in pri_train_buckets]

    return pri_w


def grid_search(y, raw_x, filename = "grid_results.csv", bucket_error_function = compute_ridge_fail_rate):

    """Calcuates the fail rate for differente values of Lambda and Gamma and prints the fail rate for each bucket and lambda with a regular regression"""

    with open(filename, 'w') as file:
        file.write("lambda,degree,A error,B error,C error,D error\n")
        degrees = range(1, 8)
        lambdas = np.logspace(-5, 0, 100)
        testing_errors = {}
        for lambda_ in tqdm(lambdas):
            for degree in tqdm(degrees):
                errors = np.array(bucket_error_function(y, raw_x, lambda_, degree))
                l = [lambda_, degree]
                file.write(",".join(str(x) for x in l))
                file.write(",")
                file.write(",".join(str(x) for x in errors))
                file.write("\n")
                