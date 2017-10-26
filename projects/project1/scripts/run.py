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
    return np.array([remove_undef(c) for i, c in enumerate(data.T) if i not in [14, 15, 17, 18, 22, 24, 25, 27, 28]])

def analyse_data(data):
    cols = columns(data)
    pos_cols = [i for i, c in enumerate(cols) if c.min() > 0]
    nez_cols = [i for i, c in enumerate(cols) if 0 not in c]
    return (pos_cols, nez_cols)

def prepare_data(data, analysed, degree = 7):
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


def grid_search(y, raw_x):
    degrees = range(1, 9)
    lambdas = np.logspace(-4, 0, 30)
    testing_errors = {}
    for lambda_ in tqdm(lambdas):
        for degree in degrees:
            errors = np.array(compute_ridge_rmse(y, raw_x, lambda_, degree))
            testing_errors[(lambda_, degree)] = np.mean(errors)
            
    return sorted(testing_errors, key=testing_errors.get)

if __name__ == "__main__":
    
    yb, raw_data, _ = helpers.load_csv_data("../data/train.csv", False)

    print(compute_ridge_rmse(yb, raw_data, 0.0001, 8))

    #compute_rmse(yb, raw_data, 0.001, 7)
    
    """ success = 0
        total = 0
        for i, ev in enumerate(test_x):
            pri = int(raw_test_x[i][22])
            x = pri_w[pri].dot(ev)

            success += (-1 if x < 0 else 1) == test_y[i]

        print("\ntotal average =", (success / len(test_x) * 100))"""


    
    """_, raw_test_data, ids = helpers.load_csv_data("../data/test.csv", True)
    test_data = prepare_data(raw_test_data)

    preds = np.ones(len(test_data))
    for i, ev in tqdm(enumerate(test_data)):
        pri = int(raw_test_data[i][22])
        x = pri_w[pri].dot(ev)
        preds[i] = -1 if x < 0 else 1

    helpers.create_csv_submission(ids, preds, "results.csv")"""


