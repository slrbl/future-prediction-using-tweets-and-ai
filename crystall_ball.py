# common imports
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go

from helpers import *

ARGS = get_args()
language = ARGS["language"] if ARGS["language"] else "en_all"
model = ARGS["model"].lower()
if model not in ["lstm", "fbprophet"]:
    print("> The only available model options are LSTM or FBProphet")

if model == "lstm":
    future_depth = int(ARGS["future_depth"]) if ARGS["future_depth"] else 10
    # Use n_steps last days to predict the current day
    n_steps = int(ARGS["steps"]) if ARGS["steps"] else 60
    EPSILON = 10 ** -5
    N_FEATURES = 1
    error_sum = 0
    errors, predictions, predictions_dates = [], [], []
    prediction_count = 0
    display_error = False

    train_from_year, train_from_month, train_from_day = decompose_date(
        ARGS["train_from"]
    )
    train_until_year, train_until_month, train_until_day = decompose_date(
        ARGS["train_until"]
    )
    from_date_fomatted = "{}-{}-{}".format(
        train_from_year, train_from_month, train_from_day
    )

    # Get data
    data = get_time_series(language, "100000", "simple", from_date_fomatted)

    df = pd.DataFrame(data["objects"])
    df["date"] = pd.to_datetime(df.date)
    df.sort_values("date", inplace=True)
    df["happiness"] = df["happiness"].astype(float)

    train_until_date = pd.datetime(train_until_year, train_until_month, train_until_day)
    prediction_date = train_until_date + pd.to_timedelta(1, unit="d")

    # Training data
    training_data = df.loc[df["date"] <= train_until_date]
    training_scores = np.array(training_data["happiness"])
    training_dates = np.array(training_data["date"])

    # Testing data
    validation_data = df.loc[df["date"] > train_until_date]
    validation_scores = np.array(validation_data["happiness"])
    validation_dates = np.array(validation_data["date"])

    # Train and Predict
    from keras.models import Sequential
    from keras.layers import LSTM
    from keras.layers import Dense

    while prediction_count < future_depth:
        print("-" * 50 + " prediction_count = {}".format(prediction_count))
        size = training_scores.shape[0]
        # Train model with training data
        X, y = split_sequence(training_scores, n_steps)
        X = X.reshape((X.shape[0], X.shape[1], N_FEATURES))
        # Define NN
        model = Sequential()
        model.add(LSTM(20, activation="relu", input_shape=(n_steps, N_FEATURES)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        # Fit model
        model.fit(X, y, epochs=200, verbose=0)
        # Get last n_steps point as the features of the new next point to predict
        next_point_steps = training_scores[size - n_steps :]
        x_input, none = split_sequence(next_point_steps, n_steps)
        x_input = x_input.reshape((1, n_steps, N_FEATURES))
        yhat = model.predict(x_input, verbose=0)[0][0]
        print("Date: {}".format(prediction_date))
        print("Prediction: {}".format(yhat))
        if prediction_count < len(validation_scores):  # .indexOf(prediction_count)>-1:
            display_error = True
            error = abs(validation_scores[prediction_count] - yhat)
            errors.append(error)
            error_sum += error
            print("Current Error: {}".format(error))
            print("Mean Error: {}".format(error_sum / (prediction_count + EPSILON)))
            # Add the new prediction to training date
            predictions_dates.append(validation_dates[prediction_count])
            training_dates = np.append(
                training_dates, validation_dates[prediction_count]
            )
        else:
            errors.append(0)
            predictions_dates.append(np.datetime64(prediction_date))
            training_dates = np.append(training_dates, np.datetime64(prediction_date))
            prediction_date += pd.to_timedelta(1, unit="d")

        predictions.append(yhat)
        training_scores = np.append(training_scores, yhat)
        prediction_count += 1

    # Plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=from_np_datetime_to_str(validation_dates),
            y=validation_scores,
            mode="lines",
            line=dict(color="blue", width=2),
            name="Actual",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=from_np_datetime_to_str(training_dates),
            y=training_scores,
            line=dict(color="green", width=2),
            mode="lines",
            name="Training",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=from_np_datetime_to_str(predictions_dates),
            y=predictions,
            line=dict(color="yellow", width=2),
            mode="lines",
            name="Prediction",
        )
    )
    if display_error:
        fig.add_trace(
            go.Scatter(
                x=from_np_datetime_to_str(predictions_dates),
                y=errors,
                line=dict(color="red", width=2),
                mode="lines+markers",
                name="Error",
            )
        )
    fig.update_layout(
        hovermode="x unified",
        title="Training, validation, prediction and error plot using LSTM model",
    )
    fig.show()

elif model == "fbprophet":
    from fbprophet import Prophet

    initial_date = ARGS["train_from"].replace("/", "-")
    train_until_datetime = datetime.strptime(ARGS["train_until"], "%Y/%m/%d")
    # depth from train_until date point in days
    future_depth = int(ARGS["future_depth"]) if ARGS["future_depth"] else 365 * 2

    data = get_time_series(language, "100000", "simple", initial_date)

    # only useful data to our study
    useful_data = [
        {"date": data_point["date"], "happiness": data_point["happiness"]}
        for data_point in data["objects"]
    ]
    df_records = pd.DataFrame.from_records(useful_data)
    df_records["date"] = pd.to_datetime(df_records.date)
    df_records.sort_values("date", inplace=True)
    df_records["happiness"] = df_records["happiness"].astype(float)

    # training and validation datasets
    df_train = df_records[df_records["date"] <= train_until_datetime]
    # only target the requested depth
    df_validation = df_records[
        (df_records["date"] > train_until_datetime)
        & (df_records["date"] < (train_until_datetime + timedelta(days=future_depth)))
    ]
    # prepare df_train to prophet column names convention
    df_train = df_train.rename(columns={"date": "ds", "happiness": "y"})

    # model
    # use multiplicative seasonality_mode given the non constant
    # seasonal effect observed on the dataset
    # use only weekly and yearly seasonality given our daily dataset
    prophet = Prophet(
        seasonality_mode="multiplicative",
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    trained_model = prophet.fit(df_train)
    build_forecast = trained_model.make_future_dataframe(periods=future_depth, freq="D")
    forecast = prophet.predict(build_forecast)

    # plot ground truth vs prediction
    forecast_validation = forecast[["ds", "yhat"]][
        forecast["ds"] > train_until_datetime
    ].rename(columns={"ds": "date", "yhat": "happiness"})
    forecast_validation["from"] = "fbprophet"
    df_validation["from"] = "ground_truth"
    df_comparison = pd.concat([df_validation, forecast_validation]).fillna("None")

    # compute error indicators
    df_validation = df_validation.set_index("date").drop(["from"], axis=1)
    forecast_validation = forecast_validation.set_index("date").drop(["from"], axis=1)
    df = df_validation.join(
        forecast_validation, lsuffix="_ground_truth", rsuffix="_predict"
    )
    df["error"] = df["happiness_ground_truth"] - df["happiness_predict"]
    df["square_error"] = df["error"].apply(lambda x: x * x)
    mse = round(df["square_error"].sum() / len(df) * 100, 2)
    df = df[df["happiness_predict"].notna()]
    mape = mape(df.happiness_ground_truth, df.happiness_predict)
    fig = go.Figure()
    # ground truth
    fig.add_trace(
        go.Scatter(
            x=df_comparison[df_comparison["from"] == "ground_truth"]["date"],
            y=df_comparison[df_comparison["from"] == "ground_truth"]["happiness"],
            mode="lines",
            line=dict(color="green", width=2),
            name="ground truth",
        )
    )

    # fb prophet prediction
    fig.add_trace(
        go.Scatter(
            x=df_comparison[df_comparison["from"] == "fbprophet"]["date"],
            y=df_comparison[df_comparison["from"] == "fbprophet"]["happiness"],
            line=dict(color="royalblue", width=2),
            mode="lines",
            name="fbprophet",
        )
    )
    # error
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["error"],
            line=dict(color="red", width=2),
            mode="lines+markers",
            name="Error",
        )
    )
    fig.update_layout(
        hovermode="x unified",
        title=f"validation, prediction and error plot using FBProphet model | MAPE {mape} %| MSE {mse} %",
    )
    fig.show()
