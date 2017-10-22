#!/usr/bin/env python3

import numpy as np


#Return a list of tuple (column index, pearson coefficient)
def pearson_coeffs(y, tX, threshold=0):
    l = [];

    for i, col in enumerate(tX.T) :
        l.append([i, np.corrcoef([y, col])[0][1]])

    return [x for x in l if abs(x[1]) >= threshold]
#    for e in sorted_l:
#        print("{} : {}".format(e[0], e[1]))

#Returns a stripped version of the data where we ignore data with pearson coefficients under the treshold
#First element is the indexes of columns kept, the second element is the stripped data
def build_pearson_coeffs_data(y, tX, threshold=0):

    coeffs = np.array(pearson_coeffs(y, tX, threshold)).T.astype(np.int64);

    _tX = tX.T[coeffs[0]].T

    return (coeffs[0], _tX)
