from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql.functions import pandas_udf, col, when, lit


#  Count the number of optional services enabled, like streaming TV
def compute_service_features(inputDF: SparkDataFrame) -> SparkDataFrame:
  # Create pandas UDF function
  @pandas_udf('double')
  def num_optional_services(*cols):
    # Nested helper function to count the number of optional services in a pandas dataframe
    return sum(map(lambda s: (s == "Yes").astype('double'), cols))

  return inputDF.\
    withColumn("num_optional_services",
        num_optional_services("online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"))