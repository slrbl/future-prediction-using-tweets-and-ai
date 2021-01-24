import argparse

import numpy as np
import requests


# Get user argument
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--train_from",
        help='Training data from date in "yyyy/mm/dd" format',
        required=True,
    )
    parser.add_argument(
        "-u",
        "--train_until",
        help='Training data until date in "yyyy/mm/dd" format',
        required=True,
    )
    parser.add_argument(
        "-d", "--future_depth", help="Number of days to predict", required=False
    )
    parser.add_argument("-s", "--steps", help="Number of steps", required=False)
    parser.add_argument("-m", "--model", help="Choose LSTM or FBProphet", required=True)
    parser.add_argument(
        "-l",
        "--language",
        help="Language. Available languages are ar_all, de_all, en_all, es_all, fr_all, id_all, ko_all, pt_all, ru_all ",
        required=False,
    )
    return vars(parser.parse_args())


# Get Data from "http://hedonometer.org"
def get_time_series(language, limit, detail_level, from_date):
    URI = "http://hedonometer.org/api/v1/"
    if detail_level == "simple":
        URI += "happiness"
    else:
        URI += "events"
    return requests.get(
        URI,
        params={
            "format": "json",
            "timeseries__title": language,
            "date__gte": from_date,
            "limit": "100000",
        },
    ).json()


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    if len(sequence) == n_steps:
        return sequence, None
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# Convert string dates to yyyy, mm, dd integers
def decompose_date(str_date):
    return [int(x) for x in str_date.split("/")]


# Mean Abs Percentage Error
def mape(ground_truth, prediction):

    ground_truth, prediction = np.array(ground_truth), np.array(prediction)
    return round(np.mean((np.abs(ground_truth - prediction) / ground_truth)) * 100, 2)


def from_np_datetime_to_str(np_datetime_list):
    return [str(dt) for dt in np_datetime_list]
