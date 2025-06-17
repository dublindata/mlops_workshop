# Databricks notebook source
# DBTITLE 1,install dbldatagen for generating the synthetic data
# MAGIC %pip install dbldatagen
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,define the working directory
import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

import mlflow
import pandas as pd
import random
import re
#remove warnings for nicer display
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("mlflow").setLevel(logging.ERROR)
from mlflow import MlflowClient
from pyspark.sql.functions import expr


# COMMAND ----------

# DBTITLE 1,Defining parameters we're going to set in YAML file
dbutils.widgets.text("catalog_use", "datascience_dev", label="Catalog to Use")
dbutils.widgets.text("schema_use", "main", label="Schema to Use")

# COMMAND ----------

catalog_use = dbutils.widgets.get("catalog_use")
schema_use = dbutils.widgets.get("schema_use")

# COMMAND ----------

spark.sql(f"use catalog {catalog_use}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_use}")
spark.sql(f"USE {catalog_use}.{schema_use}")

# COMMAND ----------

# MAGIC %sql
# MAGIC select current_catalog(), current_schema();

# COMMAND ----------

# DBTITLE 1,Defining parameters we're going to set in YAML file
# A Hive-registered Delta table containing the input data.
dbutils.widgets.text(
    "bronze_table_name",
    f"{catalog_use}.{schema_use}.advanced_churn_bronze_customers",
    label="Raw Bronze Table Name",
)
# Feature table to store the computed features.
dbutils.widgets.text(
    "inference_table_name",
    f"{catalog_use}.{schema_use}.advanced_churn_inference_table",
    label="Inference Table",
)

# COMMAND ----------

# DBTITLE 1,Defining parameters we're going to set in YAML file
bronze_table_name = dbutils.widgets.get("bronze_table_name")
inference_table_name = dbutils.widgets.get("inference_table_name")

# COMMAND ----------

print(f""" 
   bronze_table_name = {bronze_table_name}
   inference_table_name = {inference_table_name}   
""")

# COMMAND ----------

client = MlflowClient()

# COMMAND ----------

# DBTITLE 1,If the Bronze Data doesn't already exist, create it from the csv

if not spark.catalog.tableExists(bronze_table_name):
  import requests
  from io import StringIO
  #Dataset under apache license: https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE
  csv = requests.get("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv").text
  df = pd.read_csv(StringIO(csv), sep=",")
  def cleanup_column(pdf):
    # Clean up column names
    pdf.columns = [re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower().replace("__", "_") for name in pdf.columns]
    pdf.columns = [re.sub(r'[\(\)]', '', name).lower() for name in pdf.columns]
    pdf.columns = [re.sub(r'[ -]', '_', name).lower() for name in pdf.columns]
    return pdf.rename(columns = {'streaming_t_v': 'streaming_tv', 'customer_i_d': 'customer_id'})

  df = cleanup_column(df)
  print(f"creating `{bronze_table_name}` raw table")
  spark.createDataFrame(df).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(bronze_table_name)

# COMMAND ----------

# DBTITLE 1,generate synthetic data for inference
def generate_synthetic(inference_table, drift_type="label_drift"):
  import dbldatagen as dg
  import pyspark.sql.types
  from databricks.feature_engineering import FeatureEngineeringClient
  import pyspark.sql.functions as F
  from datetime import datetime, timedelta
  # Column definitions are stubs only - modify to generate correct data  
  #
  generation_spec = (
      dg.DataGenerator(sparkSession=spark, 
                      name='synthetic_data', 
                      rows=50000,
                      random=True,
                      )
      .withColumn('customer_id', 'string', template=r'dddd-AAAA')
      .withColumn('transaction_ts', 'timestamp', begin=(datetime.now() + timedelta(days=-30)), end=(datetime.now() + timedelta(days=-1)), interval="1 hour")
      .withColumn('gender', 'string', values=['Female', 'Male'], random=True, weights=[0.5, 0.5])
      .withColumn('senior_citizen', 'string', values=['No', 'Yes'], random=True, weights=[0.85, 0.15])
      .withColumn('partner', 'string', values=['No', 'Yes'], random=True, weights=[0.5, 0.5])
      .withColumn('dependents', 'string', values=['No', 'Yes'], random=True, weights=[0.7, 0.3])
      .withColumn('tenure', 'double', minValue=0.0, maxValue=72.0, step=1.0)
      .withColumn('phone_service', values=['No', 'Yes'], random=True, weights=[0.9, 0.1])
      .withColumn('multiple_lines', 'string', values=['No', 'Yes'], random=True, weights=[0.5, 0.5])
      .withColumn('internet_service', 'string', values=['Fiber optic', 'DSL', 'No'], random=True, weights=[0.5, 0.3, 0.2])
      .withColumn('online_security', 'string', values=['No', 'Yes'], random=True, weights=[0.5, 0.5])
      .withColumn('online_backup', 'string', values=['No', 'Yes'], random=True, weights=[0.5, 0.5])
      .withColumn('device_protection', 'string', values=['No', 'Yes'], random=True, weights=[0.5, 0.5])
      .withColumn('tech_support', 'string', values=['No', 'Yes'], random=True, weights=[0.5, 0.5])
      .withColumn('streaming_tv', 'string', values=['No', 'Yes', 'No internet service'], random=True, weights=[0.4, 0.4, 0.2])
      .withColumn('streaming_movies', 'string', values=['No', 'Yes', 'No internet service'], random=True, weights=[0.4, 0.4, 0.2])
      .withColumn('contract', 'string', values=['Month-to-month', 'One year','Two year'], random=True, weights=[0.5, 0.25, 0.25])
      .withColumn('paperless_billing', 'string', values=['No', 'Yes'], random=True, weights=[0.6, 0.4])
      .withColumn('payment_method', 'string', values=['Credit card (automatic)', 'Mailed check',
  'Bank transfer (automatic)', 'Electronic check'], weights=[0.2, 0.2, 0.2, 0.4])
      .withColumn('monthly_charges', 'double', minValue=18.0, maxValue=118.0, step=0.5)
      .withColumn('total_charges', 'double', minValue=0.0, maxValue=8684.0, step=20)
      .withColumn('num_optional_services', 'double', minValue=0.0, maxValue=6.0, step=1)
      .withColumn('avg_price_increase', 'float', minValue=-19.0, maxValue=130.0, step=20)
      .withColumn('churn', 'string', values=['Yes'], random=True)
      )


  # Generate Synthetic Data
  df_synthetic_data = generation_spec.build()
  df_synthetic_data.write.mode("overwrite").saveAsTable(inference_table)

generate_synthetic(inference_table=inference_table_name)
