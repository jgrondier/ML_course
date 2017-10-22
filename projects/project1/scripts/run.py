from collections import defaultdict
from implementations import *
import proj1_helpers as helpers
import numpy as np


def bucket_events(data):
	"""bucket events by PRI_jet_num"""
	return [(np.where(data[:, 24] == i)) for i in range(0, 4)]	


def remove_undef(data):
	mean = data[data != -999].mean()
	data[np.where(data == -999)] = mean
	return data

def prepare_data(data):
	"""transforms mass and filters columns"""
	cols = np.array([remove_undef(c) for c in data.T])
	return np.c_[np.ones(len(data)), 
				 np.where(data[:, 0] < 0, -1, 1), # mass bool
				 cols.T]



def linreg(y, tx):
	"""performs basic linear regression"""
	assert(len(y) == len(tx))
	assert(len(y) > 0)

	success = 0

	tx = standardize(tx)

	losses, ws = gradient_descent(y, tx, np.zeros(tx.shape[1]), 5000, 0.005)
	min_loss = min(losses)

	return next(w for l, w in zip(losses, ws) if l == min_loss)



def bench(w, y, tx):
	success = 0
	for r, ev in zip(y, standardize(tx)):
		#predictions = [-1 if we.dot(ev) < 0 else 1 for we in w]
		#prediction = -1 if np.sum(predictions) < 0 else 1
		prediction = -1 if w.dot(ev) < 0 else 1
		success += 1 if prediction == int(r) else 0

	print("  success rate =", (success / float(len(y)) * 100), "\n")
	return success



if __name__ == "__main__":
    
    yb, input_data, ids = helpers.load_csv_data("../data/small.csv", True)
    input_data = prepare_data(input_data)

    success = 0
    for i, b in enumerate(bucket_events(input_data)):
        print("Group", ["a:", "b:", "c:", "d:"][i], "(" + str(len(b[0])), "items)")
        group_w = linreg(yb[b], input_data[b])
        success += bench(group_w, yb[b], input_data[b])

    print("\ntotal average =", (success / len(yb) * 100))


