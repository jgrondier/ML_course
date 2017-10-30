from collections import defaultdict
from implementations import *
from jgrondier_helpers import *
import proj1_helpers as helpers
import numpy as np
import gc


def uniq_count(t):
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
    return np.array([remove_undef(c) for i, c in enumerate(data.T) if i not in [22]])#[14, 15, 17, 18, 22, 24, 25, 27, 28]])
                    
def analyse_data(data):
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
    analysed = analyse_data(raw_data)

    train_x = prepare_data(raw_data, analysed, degree)

    pri_train_buckets = bucket_events(raw_data)
    pri_w = [ridge_regression(yb[b], train_x[b], lambda_) for b in pri_train_buckets]

    return pri_w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, compute_loss = calculate_loss):

    threshold = 1e-8

    """Gradient descent algorithm."""
    losses = []
    w = initial_w

    for n_iter in tqdm(range(max_iters)):
        loss, g, _ = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient

        losses.append(loss)

        if n_iter > 0 and np.abs(losses[n_iter] - losses[n_iter-1]) < threshold:
            return loss, w

    return losses[-1], w


def grid_search(y, raw_x, filename = "grid_results.csv", bucket_error_function = compute_ridge_fail_rate):
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
                
def grid_search_logit(y, raw_x, filename = "logit_grid_results.csv"):
    with open(filename, 'w') as file:
        file.write("lambda,degree,gamma,threshhold,A error,B error,C error,D error\n")
        degrees = range(1, 8)
        lambdas = np.logspace(-5, -1, 60)
        gammas = [.017]#np.logspace(-3, -1, 30)
        thresholds = [1e-8]#np.logspace(-10, -3, 30)
        testing_errors = {}
        
        logit_y = y*0.5+0.5
        
        for lambda_ in tqdm(lambdas):
            for degree in tqdm(degrees):
                for gamma in tqdm(gammas):
                    for threshold in tqdm(thresholds):
                        errors = np.array(compute_logit_fail_rate(logit_y, raw_x, lambda_, degree, gamma, threshold))
                        l = [lambda_, degree, gamma, threshold]
                        file.write(",".join(str(x) for x in l))
                        file.write(",")
                        file.write(",".join(str(x) for x in errors))
                        file.write("\n")

if __name__ == "__main__":
    """for ridge:
    A: 1e-5, 1
    B: 2.5e-5, 7
    C: 1e-5, 2
    D: 1e-4, 7
    """

    yb, raw_data, _ = helpers.load_csv_data("../data/train.csv", False)
    #print(compute_ridge_fail_rate(yb, raw_data, 1e-5, 7))

    #grid_search(yb, raw_data)
    #print(np.mean(compute_ridge_rmse(yb, raw_data, 0.017, 6)))

    """ success = 0
        total = 0
        for i, ev in enumerate(test_x):
            pri = int(raw_test_x[i][22])
            x = pri_w[pri].dot(ev)

            success += (-1 if x < 0 else 1) == test_y[i]

        print("\ntotal average =", (success / len(test_x) * 100))"""


    #"""
    _, raw_test_data, ids = helpers.load_csv_data("../data/test.csv", False)
    lambdas = [1e-5, 2.5e-5, 1e-5, 1e-4]
    degrees = [1,7,2,7]
    
    preds = []
    ids_final = []
    
    for i in tqdm(range(len(degrees))):
        test_data = prepare_data(raw_test_data, analyse_data(raw_data), degrees[i])
        w = train_ridge_rmse(yb, raw_data, lambdas[i], degrees[i])[i]
        for idx, ev in tqdm(enumerate(test_data)):
            pri = int(raw_test_data[idx][22])
            if pri == i:
                x = w.dot(ev)
                preds.append(-1 if x < 0 else 1)
                ids_final.append(ids[idx])

    helpers.create_csv_submission(ids_final, preds, "results.csv")#"""

    """
    y_train, raw_data, _ = helpers.load_csv_data("../data/train.csv", False)
    train_data = prepare_data(raw_data, analyse_data(raw_data), 6)

    max_iter = 100
    gamma = 0.01
    lambda_ = 0.017
    threshold = 1e-8
    losses = []


    y,x = y_train, train_data
    y = np.expand_dims(y, axis=1)

    #pri_train_buckets = bucket_events(raw_data)
    #pri_w = [train_logistic(y[b], x[b], gamma, lambda_, max_iter, threshold, compute_loss_MSE)[0] for b in pri_train_buckets]

    pri_w = train_logistic(y, x, gamma, lambda_, max_iter, threshold, calculate_loss)[0]
    
    _, raw_test_data, ids = helpers.load_csv_data("../data/test.csv", False)
    test_data = prepare_data(raw_test_data, analyse_data(raw_data), 6)#"""

    """preds = np.ones(len(test_data))
    for i, ev in tqdm(enumerate(test_data)):
        pri = int(raw_test_data[i][22])
        z = pri_w[pri][:,0].dot(ev)
        preds[i] = -1 if z < 0 else 1"""
    """
    preds = np.ones(len(test_data))
    for i, ev in tqdm(enumerate(test_data)):
        z = pri_w[:,0].dot(ev)
        preds[i] = -1 if z < 0 else 1


    helpers.create_csv_submission(ids, preds, "results.csv")#"""
