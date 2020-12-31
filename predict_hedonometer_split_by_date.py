import requests
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

URI = "http://hedonometer.org/api/v1/happiness"

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
    return array(X), array(y)

data = requests.get(URI,params={'format': 'json','timeseries__title':'en_all','date__gte':'2014-01-01','limit':'10000'}).json()['objects']

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df.date)
df.sort_values('date', inplace=True)
df["happiness"] = df["happiness"].astype(float)

split_date = pd.datetime(2021,1,1)

# Testing data
validation_data = df.loc[df['date'] >= split_date]
validation_scores = np.array(validation_data["happiness"])
validation_dates = np.array(validation_data["date"])

# Training data
training_data = df.loc[df['date'] < split_date]
training_scores = np.array(training_data["happiness"])
training_dates = np.array(training_data["date"])

n_steps = 60
future_depth = 60
n_features = 1
error_sum = 0

errors = []
predictions = []
predictions_dates = []

prediction_count = 0
new_date = split_date

while prediction_count < future_depth:
    print ("*"*100+" prediction_count ="+str(prediction_count))
    size = training_scores.shape[0]
    # Train model with training data
    X, y = split_sequence(training_scores, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(20, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=200, verbose=0)
    # Get last n_steps point as the features of the new next point to predict
    next_point_steps = training_scores[size-n_steps:]
    x_input, none = split_sequence(next_point_steps, n_steps)
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose = 0)[0][0]
    if prediction_count<len(validation_scores):#.indexOf(prediction_count)>-1:
        error = abs(validation_scores[prediction_count] - yhat)
        errors.append(error)
        error_sum += error
        print("PREDICTION:"+str(yhat))
        print("CURRENT ERROR:"+str(error))
        print("MEAN ERROR:"+str(error_sum/(prediction_count+0.0000000001)))
        # Add the new prediction to training date
        predictions_dates.append(validation_dates[prediction_count])
        training_dates = np.append(training_dates,validation_dates[prediction_count])
        print(type(validation_dates[prediction_count]))
    else:
        errors.append(0)
        predictions_dates.append(np.datetime64(new_date))
        training_dates = np.append(training_dates,np.datetime64(new_date))
        new_date += pd.to_timedelta(1,unit='d')

    predictions.append(yhat)
    training_scores = np.append(training_scores,yhat)
    #training_dates = np.append(training_dates,validation_dates[prediction_count])
    prediction_count += 1

errors = np.array(errors)
predictions = np.array(predictions)
predictions_dates = np.array(predictions_dates)
print(training_dates)
# The division by 10 goal is to be able to see error in the same chart
validation_scores = validation_scores/10
training_scores = training_scores/10
predictions = predictions/10
plt.plot(validation_dates, validation_scores, 'b', label = 'Actual')
plt.plot(training_dates, training_scores, 'g', label = 'Training')
plt.plot(predictions_dates, predictions, 'y', label = 'Prediction')

plt.plot(predictions_dates, errors, 'r', label = 'Error')

plt.legend(loc = "best")
plt.show()
