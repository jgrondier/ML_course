from collections import defaultdict
from implementations import *
import proj1_helpers as helpers
import numpy as np


yb, input_data, ids = helpers.load_csv_data("../data/small.csv", True)


def bucket_events(data):
	"""bucket events by PRI_jet_num"""
	return [(np.where(data[:, 22] == i)) for i in range(0, 4)]

def prepare_data(data):
	"""transforms mass into a bool [-1; 1] and filters columns"""
	data[:, 0] = np.where(data[:, 0] < 0, -1, 1)
	x = standardize(data[:, data[0, :] != -999])
	return np.c_[np.ones(len(x)), x]





# test on first dataset
indexes = bucket_events(input_data)[0]

y = yb[indexes]

tx = prepare_data(input_data[indexes])
w = np.zeros(tx.shape[1])

losses, ws = gradient_descent(y, tx, w, 5000, 0.01)
min_loss = min(losses)

print([w for l, w in zip(losses, ws) if l == min_loss])
