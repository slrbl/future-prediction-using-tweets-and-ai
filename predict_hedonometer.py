# Author: walid.daboubi@gmail.com
# About: Predict big events using Hedonometer.org data
# Version:
    # v1 2020/12/31
    # v1.1 2021/01/01

from helpers import *

URI = "http://hedonometer.org/api/v1/happiness"
EPSILON = 10**-5

ARGS = get_args()

future_depth = 10
if ARGS['future_depth'] != None:future_depth = int(ARGS['future_depth'])
# Use n_steps last days to predit the current day
n_steps = 60
if ARGS['steps'] != None:n_steps = int(ARGS['steps'])
language = 'en_all'
if ARGS['language'] != None:language = ARGS['language']
n_features = 1
error_sum = 0
errors, predictions, predictions_dates = [], [], []
prediction_count = 0
display_error = False

train_from_year, train_from_month, train_from_day = decompose_date(ARGS['train_from'])
train_until_year, train_until_month, train_until_day = decompose_date(ARGS['train_until'])
from_date_fomatted = '{}-{}-{}'.format(train_from_year, train_from_month, train_from_day)

# Get data
data = requests.get(URI, params = {'format': 'json','timeseries__title':language,'date__gte':from_date_fomatted,'limit':'100000'}).json()

df = pd.DataFrame(data['objects'])
df['date'] = pd.to_datetime(df.date)
df.sort_values('date', inplace=True)
df["happiness"] = df["happiness"].astype(float)

train_until_date = pd.datetime(train_until_year, train_until_month, train_until_day)
prediction_date = train_until_date + pd.to_timedelta(1, unit = 'd')

# Training data
training_data = df.loc[df['date'] <= train_until_date]
training_scores = np.array(training_data["happiness"])
training_dates = np.array(training_data["date"])

# Testing data
validation_data = df.loc[df['date'] > train_until_date]
validation_scores = np.array(validation_data["happiness"])
validation_dates = np.array(validation_data["date"])


# Train and Predict
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Dense
while prediction_count < future_depth:
    print ('-'*50+' prediction_count = {}'.format(prediction_count))
    size = training_scores.shape[0]
    # Train model with training data
    X, y = split_sequence(training_scores, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # Define NN
    model = Sequential()
    model.add(LSTM(20, activation = 'relu', input_shape = (n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer = 'adam', loss='mse')
    # Fit model
    model.fit(X, y, epochs=200, verbose=0)
    # Get last n_steps point as the features of the new next point to predict
    next_point_steps = training_scores[size-n_steps:]
    x_input, none = split_sequence(next_point_steps, n_steps)
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose = 0)[0][0]
    print('Date: {}'.format(prediction_date))
    print('Prediction: {}'.format(yhat))
    if prediction_count<len(validation_scores):#.indexOf(prediction_count)>-1:
        display_error = True
        error = abs(validation_scores[prediction_count] - yhat)
        errors.append(error)
        error_sum += error
        print('Current Error: {}'.format(error))
        print('Mean Error: {}'.format(error_sum/(prediction_count+EPSILON)))
        # Add the new prediction to training date
        predictions_dates.append(validation_dates[prediction_count])
        training_dates = np.append(training_dates,validation_dates[prediction_count])
    else:
        errors.append(0)
        predictions_dates.append(np.datetime64(prediction_date))
        training_dates = np.append(training_dates,np.datetime64(prediction_date))
        prediction_date += pd.to_timedelta(1,unit='d')

    predictions.append(yhat)
    training_scores = np.append(training_scores,yhat)
    prediction_count += 1

# Convert arrays to np.array
errors = np.array(errors)
predictions = np.array(predictions)
predictions_dates = np.array(predictions_dates)

# The division by 10 goal is to be able to see error in the same chart
validation_scores, training_scores, predictions = validation_scores/10, training_scores/10, predictions/10

# Plot
plt.plot(validation_dates, validation_scores, 'b', label = 'Actual')
plt.plot(training_dates, training_scores, 'g', label = 'Training')
plt.plot(predictions_dates, predictions, 'y', label = 'Prediction')
if display_error == True:
    plt.plot(predictions_dates, errors, 'r', label = 'Error')
plt.legend(loc = "best")
plt.show()
