from collections import defaultdict
import proj1_helpers as helpers
import numpy as np



yb, input_data, ids = helpers.load_csv_data("../data/train.csv", True)



def event_mask(ev):
	"""integer value representing the category of the event"""
	return sum((v != -999) << j for j, v in enumerate(ev))

def bucket_events(data):
	"""buckets events by category: returns a list containing list of indexes for each category"""
	d = defaultdict(list)
	for i, ev in enumerate(data):
		d[event_mask(ev)].append(i)
	return d.values()


print(len(bucket_events(input_data)))