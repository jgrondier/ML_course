from collections import defaultdict
from implementations import *
import proj1_helpers as helpers
import numpy as np


def uniq(t):
	vals = set()
	for x in t:
		vals.add(x)
	return len(vals)


def standardize_matrix(data):
	cols = np.array([standardize(c) if uniq(c) > 8 else c for c in data.T])
	return cols.T


def bucket_events(data):
	"""bucket events by PRI_jet_num"""
	return [np.where(data[:, 22] == i)[0] for i in range(0, 4)]	



def bucket_events_by_undefs(data):
	"""buckets events by category: returns a list containing list of indexes for each category"""
	def event_mask(ev):
		"""integer value representing the category of the event"""
		return sum((v != -999) << j for j, v in enumerate(ev))

	d = defaultdict(list)
	for i, ev in enumerate(data):
		d[event_mask(ev)].append(i)
	return list(d.values())


def remove_undef(data):
	mean = data[data != -999].mean()
	data[np.where(data == -999)] = mean
	return data


def prepare_data(data):
	"""transforms mass and filters columns"""
	cols = np.array([remove_undef(c) for c in np.copy(data.T)])
	return np.c_[np.ones(len(data)), 
				 np.where(data[:, 0] < 0, -1, 1), # mass bool
				 standardize_matrix(cols.T)]




def linreg(y, tx):
	"""performs basic linear regression"""
	assert(len(y) == len(tx))
	assert(len(y) > 0)

	success = 0

	losses, ws = gradient_descent(y, tx, np.zeros(tx.shape[1]), 5000, 0.01)
	min_loss = min(losses)

	return next(w for l, w in zip(losses, ws) if l == min_loss)




if __name__ == "__main__":
    
    yb, raw_data, ids = helpers.load_csv_data("../data/train.csv", True)

    data = prepare_data(raw_data)
    print(data.shape)

    #global_w = linreg(yb, data)

    pri_buckets = bucket_events(raw_data)
    #undef_buckets = bucket_events_by_undefs(raw_data)

    pri_jet_w = [linreg(yb[b], data[b]) for b in pri_buckets]
    #undef_w = [linreg(yb[b], data[b]) for b in undef_buckets]

    success = 0
    for i, ev in tqdm(enumerate(data)):
    	p_w = pri_jet_w[int(raw_data[i][22])];

    	x = p_w.dot(ev)
    	prediction = -1 if x < 0 else 1

    	success += int(prediction) == int(yb[i])
    	

    print("\ntotal average =", (success / len(yb) * 100))


