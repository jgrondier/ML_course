from collections import defaultdict
import proj1_helpers as helpers
import numpy as np



yb, input_data, ids = helpers.load_csv_data("../data/train.csv", True)


def bucket_events(data):
	"""bucket events by PRI_jet_num"""
	return [(np.where(input_data[:, 22] == i)) for i in range(0, 4)]

def prepare_data(data):
	"""transforms mass into a bool [-1; 1] and filters columns"""
	data[:, 0] = np.where(data[:, 0] < 0, -1, 1)
	return data[:, data[0, :] != -999]





# test on first dataset
indexes = bucket_events(input_data)[0]

y = yb[indexes]
x = prepare_data(input_data[indexes])