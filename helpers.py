# Author: walid.daboubi@gmail.com
# About: Predict big events using Hedonometer.org data
# Version:
    # v1 2020/12/31
    # v1.1 2021/01/01
    
import argparse
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get user argument 
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--train_from', help = 'Training data from date in "yyyy/mm/dd" format', required = True)
    parser.add_argument('-u', '--train_until', help = 'Training data until date in "yyyy/mm/dd" format', required = True)
    parser.add_argument('-d', '--future_depth', help = 'Number of days to predict', required = False)
    parser.add_argument('-s', '--steps', help = 'Number of steps', required = False)
    parser.add_argument('-l', '--language', help = 'Language. Available languages are ar_all, de_all, en_all, es_all, fr_all, id_all, ko_all, pt_all, ru_all ', required = False)
    return vars(parser.parse_args())

# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    if len(sequence) == n_steps:
        return sequence,None
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

# Convert string dates to yyyy, mm, dd integers 
def decompose_date(str_date):
    return [int(x) for x in str_date.split('/')]

# Mean Abs Percentage Error
def mape(ground_truth, prediction):

    ground_truth,prediction = np.array(ground_truth),np.array(prediction)
    return round(np.mean((np.abs(ground_truth-prediction)/ground_truth))*100,2)
