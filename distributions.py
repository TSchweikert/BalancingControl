#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 10:43:23 2021

@author: sarah
"""
import numpy as np
from scipy.special import beta as beta_func
from scipy.special import gamma as gamma_func
import matplotlib.pylab as plt

def Beta(x, alpha=1, beta=1):
    
    # x must be >0 and <1!
    dist = np.power(x, alpha-1) * np.power(1-x, beta-1)
    dist /= beta_func(alpha, beta)
    
    return dist


def Gamma(x, concentration=1, rate=1):
    
    exp = np.exp(-rate*x)
    dist = np.power(x, concentration-1) * exp
    
    dist *= np.power(rate, concentration)
    dist /= gamma_func(concentration)
    
    return dist
    