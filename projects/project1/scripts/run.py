from collections import defaultdict
from implementations import *
import proj1_helpers as helpers
import numpy as np


def bucket_events(data):
	"""bucket events by PRI_jet_num"""
	return [(np.where(data[:, 22] == i)) for i in range(0, 4)]	

def prepare_data(data):
	"""transforms mass and filters columns"""
	data[:, 0] = np.where(data[:, 0] < 0, -1, 1)
	x = standardize(data[:, data[0, :] != -999])
	return np.c_[np.ones(len(x)), x]




def linreg(y, tx):
	"""performs basic linear regression"""
	assert(len(y) == len(tx))

	success = 0

	tx = prepare_data(tx)
	losses, ws = gradient_descent(y, tx, np.zeros(tx.shape[1]), 5000, 0.01)
	min_loss = min(losses)
	best_w = next(w for l, w in zip(losses, ws) if l == min_loss)

	for r, ev in zip(y, tx):
		prediction = -1 if best_w.dot(ev) < 0 else 1
		success += 1 if prediction == int(r) else 0

	print("  success rate =", (success / float(len(y)) * 100), "\n")

	return best_w, success


if __name__ == "__main__":
    
    yb, input_data, ids = helpers.load_csv_data("../data/small.csv", True)

    no_mass = input_data[:, 0] == -999
    no_mass_y = yb[no_mass]
    no_mass_data = input_data[no_mass]

    print("Without mass:")
    _, success = linreg(no_mass_y, no_mass_data)

    with_mass = input_data[:, 0] != -999
    mass_y = yb[with_mass]
    mass_data = input_data[with_mass]

    for i, b in enumerate(bucket_events(mass_data)):
        print("Group", ["a:", "b:", "c:", "d:"][i])
        _, s = linreg(mass_y[b], mass_data[b])
        success += s

    print("\ntotal =", (success / len(yb) * 100))


