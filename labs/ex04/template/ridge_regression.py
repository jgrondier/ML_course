# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    txtxt = tx.T.dot(tx)
    lambda_identity = (2*len(y)*lambda_)*np.identity(len(txtxt))
    first = txtxt + lambda_identity
    firstinv = np.linalg.inv(first)
    firstinvdottxt = firstinv.dot(tx.T)
    return firstinvdottxt.dot(y)
