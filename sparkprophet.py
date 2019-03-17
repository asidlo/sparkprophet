"""sparkprophet contains sample code for running fbprophet on Apache Spark."""

import numpy as np
import pandas as pd
from fbprophet import Prophet
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, struct
from pyspark.sql.types import FloatType, StructField, StructType, StringType, TimestampType
from sklearn.metrics import mean_squared_error

changepoint_prior_scale = 0.005
seasonality_prior_scale = 0.05
changepoint_range = 0.5


def retrieve_data():
    """Load sample data from ./data/original-input.csv as a pyspark.sql.DataFrame."""
    df = (spark.read
          .option("header", "true")
          .option("inferSchema", value=True)
          .csv("./data/input.csv"))

    # Drop any null values incase they exist
    df = df.dropna()

    # Rename timestamp to ds and total to y for fbprophet
    df = df.select(
        df['timestamp'].alias('ds'),
        df['app'],
        df['value'].cast(FloatType()).alias('y'),
        df['metric']
    )

    return df


def transform_data(row):
    """Transform data from pyspark.sql.Row to python dict to be used in rdd."""
    data = row['data']
    app = row['app']
    mt = row['metric']

    # Transform [pyspark.sql.Dataframe.Row] -> [dict]
    data_dicts = []
    for d in data:
        data_dicts.append(d.asDict())

    # Convert into pandas dataframe for fbprophet
    data = pd.DataFrame(data_dicts)
    data['ds'] = pd.to_datetime(data['ds'])

    return {
        'app': app,
        'metric': mt,
        'data': data
    }


def partition_data(d):
    """Split data into training and testing based on timestamp."""
    # Extract data from pd.Dataframe
    data = d['data']

    # Find max timestamp and extract timestamp for start of day
    max_datetime = pd.to_datetime(max(data['ds']))
    start_datetime = max_datetime.replace(hour=00, minute=00, second=00)

    # Extract training data
    train_data = data[data['ds'] < start_datetime]

    # Account for zeros in data while still applying uniform transform
    train_data['y'] = train_data['y'].apply(lambda x: np.log(x + 1))

    # Assign train/test split
    d['test_data'] = data.loc[(data['ds'] >= start_datetime)
                              & (data['ds'] <= max_datetime)]
    d['train_data'] = train_data

    return d


def create_model(d):
    """Create Prophet model using each input grid parameter set."""
    m = Prophet(seasonality_prior_scale=seasonality_prior_scale,
                changepoint_prior_scale=changepoint_prior_scale,
                changepoint_range=changepoint_range,
                interval_width=0.95, weekly_seasonality=True, daily_seasonality=True)
    d['model'] = m

    return d


def train_model(d):
    """Fit the model using the training data."""
    model = d['model']
    train_data = d['train_data']
    model.fit(train_data)
    d['model'] = model

    return d


def test_model(d):
    """Run the forecast method on the model to make future predictions."""
    test_data = d['test_data']
    model = d['model']
    t = test_data['ds']
    t = pd.DataFrame(t)
    t.columns = ['ds']

    predictions = model.predict(t)
    d['predictions'] = predictions

    return d


def make_forecast(d):
    """Execute the forecast method on the model to make future predictions."""
    model = d['model']
    future = model.make_future_dataframe(
        periods=576, freq='5min', include_history=False)
    future = pd.DataFrame(future['ds'].apply(pd.DateOffset(1)))
    forecast = model.predict(future)
    d['forecast'] = forecast

    return d


def normalize_predictions(d):
    """Normalize predictions using np.exp()."""
    predictions = d['predictions']
    predictions['yhat'] = np.exp(predictions['yhat']) - 1
    d['predictions'] = predictions
    return d


def normalize_forecast(d):
    """Normalize predictions using np.exp().

    Note:  np.exp(>709.782) = inf, so replace value with None
    """
    forecast = d['forecast']
    forecast['yhat'] = forecast['yhat'].apply(
        lambda x: np.exp(x) - 1 if x < 709.782 else None)
    forecast['yhat_lower'] = forecast['yhat_lower'].apply(
        lambda x: np.exp(x) - 1 if x < 709.782 else None)
    forecast['yhat_upper'] = forecast['yhat_upper'].apply(
        lambda x: np.exp(x) - 1 if x < 709.782 else None)
    d['forecast'] = forecast
    return d


def calc_error(d):
    """Calculate error using mse (mean squared error)."""
    test_data = d['test_data']
    predictions = d['predictions']
    results = mean_squared_error(test_data['y'], predictions['yhat'])
    d['mse'] = results

    return d


def reduce_data_scope(d):
    """Return a tuple (app + , + metric_type, {})."""
    return (
        d['app'] + ',' + d['metric'],
        {
            'forecast': d['forecast'],
            'mse': d['mse'],
        },
    )


def expand_predictions(d):
    """Flatten rdd into tuple which will be converted into a dataframe.Row.

    Checks each float to see if it is a np datatype, since it could be None.
    If it is an np datatype then it will convert to scalar python datatype
    so that it can be persisted into a database, since most dont know how to
    interpret np python datatypes.
    """
    app_metric, data = d
    app, metric = app_metric.split(',')
    return [
        (
            app,
            metric,
            p['ds'].to_pydatetime(),
            np.asscalar(p['yhat']) if isinstance(
                p['yhat'], np.generic) else p['yhat'],
            np.asscalar(p['yhat_lower']) if isinstance(
                p['yhat_lower'], np.generic) else p['yhat_lower'],
            np.asscalar(p['yhat_upper']) if isinstance(
                p['yhat_upper'], np.generic) else p['yhat_upper'],
            np.asscalar(data['mse']) if isinstance(
                data['mse'], np.generic) else data['mse'],
        ) for i, p in data['forecast'].iterrows()
    ]


if __name__ == '__main__':
    conf = (SparkConf()
            .setMaster("local[*]")
            .setAppName("SparkFBProphet Example"))

    spark = (SparkSession
             .builder
             .config(conf=conf)
             .getOrCreate())

    # Removes some of the logging after session creation so we can still see output
    # Doesnt remove logs before/during session creation
    # To edit more logging you will need to set in log4j.properties on cluster
    sc = spark.sparkContext
    sc.setLogLevel("INFO")

    # Retrieve data from local csv datastore
    df = retrieve_data()

    # Can subset the data by uncommenting the following line and editing array
    # df = df[df.app.isin(['a'])]

    # Group data by app and metric_type to aggregate data for each app-metric combo
    df = df.groupBy('app', 'metric')
    df = df.agg(collect_list(struct('ds', 'y')).alias('data'))

    df = (df.rdd
          .map(lambda r: transform_data(r))
          .map(lambda d: partition_data(d))
          # prophet cant handle data with < 2 training examples
          .filter(lambda d: len(d['train_data']) > 2)
          .map(lambda d: create_model(d))
          .map(lambda d: train_model(d))
          .map(lambda d: test_model(d))
          .map(lambda d: make_forecast(d))
          .map(lambda d: normalize_forecast(d))
          .map(lambda d: normalize_predictions(d))
          .map(lambda d: calc_error(d))
          .map(lambda d: reduce_data_scope(d))
          .flatMap(lambda d: expand_predictions(d)))

    schema = StructType([
        StructField("app", StringType(), True),
        StructField("metric", StringType(), True),
        StructField("ds", TimestampType(), True),
        StructField("yhat", FloatType(), True),
        StructField("yhat_lower", FloatType(), True),
        StructField("yhat_upper", FloatType(), True),
        StructField("mse", FloatType(), True)
    ])

    df = spark.createDataFrame(df, schema)
    df.write.options(header=True).csv('./data/output', mode='overwrite')

    spark.stop()
