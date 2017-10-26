# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np

def compute_loss_MSE(y, tx, w):
    """Calculate the loss"""
    e = (y - tx.dot(w))
    return e.T.dot(e) / len(y)


def least_squares(y, tx):
    """calculate the least squares solution."""
    # returns mse, and optimal weights
    # opt = np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y);
    opt = np.linalg.inv(np.dot(tx.T, tx))
    opt = np.dot(opt, tx.T)
    opt = np.dot(opt, y)

    
    return compute_loss_MSE(y, tx, opt), opt
