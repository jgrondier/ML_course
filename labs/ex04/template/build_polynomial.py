# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    pol_basis = np.ones((len(x), degree+1))
    for i in range(1, degree+1):
        pol_basis[:,i] = pol_basis[:,i-1] * x
    return pol_basis
