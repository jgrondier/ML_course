# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def least_squares(y, tx):
    """calculate the least squares."""
    opt = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y);
    opt = np.linalg.lstsq(tx, y)[0]
    return compute_loss_MSE(y, tx, opt), opt
