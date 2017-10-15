from collections import defaultdict
import proj1_helpers as helpers
import numpy as np



yb, input_data, ids = helpers.load_csv_data("../data/train.csv", True)


def bucket_events(data):
	return [(np.where(input_data[:, 22] == i)) for i in range(0, 4)]


print("loaded")

# test
indexes = bucket_events(input_data)[0]

y = yb[indexes]
x = input_data[indexes][:, 1]