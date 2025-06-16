import pyspark.sql
import pytest
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession

from ny_taxi_koeppen.feature_engineering.features.dropoff_features import (
    compute_features_fn,
)

# This fixture provides a SparkSession for the test. It's scoped to the whole session (created once per test run)
@pytest.fixture(scope="session")
def spark(request):
    """fixture for creating a spark session
    Args:
        request: pytest.FixtureRequest object
    """
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("pytest-pyspark-local-testing")
        .getOrCreate()
    )
    request.addfinalizer(lambda: spark.stop())

    return spark

# This test uses the "Spark" fixture above 
@pytest.mark.usefixtures("spark")
def test_dropoff_features_fn(spark):
    # * Create a sample pandas DF with fake taxi trip data 
    input_df = pd.DataFrame(
        {
            "tpep_pickup_datetime": [datetime(2022, 1, 10)],
            "tpep_dropoff_datetime": [datetime(2022, 1, 10)],
            "dropoff_zip": [94400],
            "trip_distance": [2],
            "fare_amount": [100],
        }
    )
    # * Convert the fake trip data to a Spark DF
    spark_df = spark.createDataFrame(input_df)
    # * Call the actual feature engineering function you're testing. 
    output_df = compute_features_fn(
        spark_df, "tpep_pickup_datetime", datetime(2022, 1, 1), datetime(2022, 1, 15)
    )
    assert isinstance(output_df, pyspark.sql.DataFrame)
    assert output_df.count() == 1
