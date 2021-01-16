import requests
import pandas as pd
import plotly.express as px
from fbprophet import Prophet
from helpers import get_args, mape
from datetime import timedelta, datetime

ARGS = get_args()

initial_date = ARGS["train_from"].replace("/", "-")
train_until_datetime = datetime.strptime(ARGS["train_until"], "%Y/%m/%d")
language = ARGS["language"] if ARGS["language"] else "en_all"
# depth from train_until date point in days
future_depth = int(ARGS["future_depth"]) if ARGS["future_depth"] else 365 * 2

# raw data
URI = "http://hedonometer.org/api/v1/events"
response = requests.get(
    URI,
    params={
        "format": "json",
        "timeseries__title": language,
        "date__gte": initial_date,
        "limit": "100000",
    },
).json()
# only useful data to our study
useful_data = [
    {
        "date": data_point["happs"]["date"],
        "happiness": data_point["happs"]["happiness"],
        "event": data_point["longer"],
    }
    for data_point in response["objects"]
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
df_train = df_train.drop("event", axis=1).rename(
    columns={"date": "ds", "happiness": "y"}
)

# model
# use multiplicative seasonality_mode given the non constant
# seasonal effect observed on the dataset
# use only weekly and yearly seasonality given our daily dataset
prophet = Prophet(
    seasonality_mode="multiplicative", weekly_seasonality=True, yearly_seasonality=True
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
fig = px.line(
    df_comparison,
    x="date",
    y="happiness",
    color="from",
    hover_name="event",
    title="Happiness over time",
)
fig.show()

# plot error
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
fig = px.line(
    df,
    x=df.index,
    y="error",
    hover_name="event",
    title=f"Error over time [MSE : {mse} % / MAPE : {mape} %]",
)
fig.update_traces(line_color="red")
fig.show()

# plot trend and seasonality
prophet.plot_components(forecast)
