from collections import defaultdict
from implementations import *
from jgrondier_helpers import *
import proj1_helpers as helpers
import numpy as np



def standardize_matrix(data):
    """standardize matrix column by column"""
    def uniq_count(t):
        vals = set()
        for x in t:
            vals.add(x)
        return len(vals)

    cols = np.array([standardize(c) if uniq_count(c) > 8 else c for c in data.T])
    return cols.T


def bucket_events(data):
    """bucket events by PRI_jet_num"""
    return [np.where(data[:, 22] == i)[0] for i in range(0, 4)]    

def remove_undef(data):
    """replace -999 by the mean"""
    data = np.copy(data)
    mean = data[data != -999].mean()
    data[np.where(data == -999)] = mean
    return data


def prepare_data(data):
    """transforms mass and add derivated features"""
    cols = np.array([remove_undef(c) for c in data.T])
    pos_cols = [c for c in cols if c.min() > 0]
    nez_cols = [c for c in cols if 0 not in c]

    cols_2 = np.array([np.power(c, 2) for c in cols])
    cols_3 = np.array([np.power(c, 3) for c in cols])
    cols_4 = np.array([np.power(c, 4) for c in cols])
    cols_5 = np.array([np.power(c, 5) for c in cols])
    cols_6 = np.array([np.power(c, 6) for c in cols])
    cols_7 = np.array([np.power(c, 7) for c in cols])
    #boolified = np.array([np.where(c < 0, -1, 1) for c in cols])

    matrix = np.c_[cols.T, np.log(pos_cols).T, np.reciprocal(nez_cols).T, cols_2.T, cols_3.T, cols_4.T, cols_5.T, cols_6.T, cols_7.T]
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


if __name__ == "__main__":
    
    yb, raw_data, _ = helpers.load_csv_data("../data/train.csv", True)

    data = prepare_data(raw_data)
    print(data.shape)

    pri_buckets = bucket_events(raw_data)
    pri_w = [ridge_regression(yb[b], data[b], 0.001) for b in pri_buckets]

    success = 0
    for i, ev in tqdm(enumerate(data)):
        pri = int(raw_data[i][22])
        x = pri_w[pri].dot(ev)
        success += (-1 if x < 0 else 1) == yb[i]

    print("\ntotal average =", (success / len(yb) * 100))

    
    """_, raw_test_data, ids = helpers.load_csv_data("../data/test.csv", True)
    test_data = prepare_data(raw_test_data)

    preds = np.ones(len(test_data))
    for i, ev in tqdm(enumerate(test_data)):
        pri = int(raw_test_data[i][22])
        x = pri_jet_w[pri];.dot(ev)
        preds[i] = -1 if x < 0 else 1

    helpers.create_csv_submission(ids, preds, "results.csv")"""


