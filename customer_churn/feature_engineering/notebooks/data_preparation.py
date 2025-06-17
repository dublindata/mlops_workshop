# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction Data Preparation
# MAGIC Our first step is to analyze the data and build the features we'll use to train our model. Let's see how this can be done.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/mlops-uc-end2end-1.png?raw=true" width="1200">
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable the collection or disable the tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=984752964297111&notebook=%2F01-mlops-quickstart%2F01_feature_engineering&demo_name=mlops-end2end&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fmlops-end2end%2F01-mlops-quickstart%2F01_feature_engineering&version=1&user_hash=a3692eff9e5299c6a85c26f2dc27b2e2000517102cea778a7cc80efff9afb355">

# COMMAND ----------

# MAGIC %pip install --quiet mlflow==2.19 databricks-feature-engineering==0.8.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from datetime import datetime, timedelta
from pyspark.sql.functions import lit
import pyspark.sql.functions as F

# COMMAND ----------

# MAGIC %md
# MAGIC # Define Variables

# COMMAND ----------

dbutils.widgets.text("catalog_use", "datascience_dev", label="Catalog to Use")
dbutils.widgets.text("schema_use", "main", label="Schema to Use")

# COMMAND ----------

catalog_use = dbutils.widgets.get("catalog_use")
schema_use = dbutils.widgets.get("schema_use")
spark.sql(f"USE {catalog_use}.{schema_use}")

# COMMAND ----------

# MAGIC %sql
# MAGIC select current_catalog(), current_schema();

# COMMAND ----------

# A Hive-registered Delta table containing the input data.
dbutils.widgets.text(
    "input_table_name",
    f"{catalog_use}.{schema_use}.advanced_churn_bronze_customers",
    label="Input Table Name",
)
# Feature table to store the computed features.
dbutils.widgets.text(
    "advanced_churn_label_table",
    f"{catalog_use}.{schema_use}.advanced_churn_label_table",
    label="Label Table",
)

# Feature table to store the computed features.
dbutils.widgets.text(
    "advanced_churn_feature_table",
    f"{catalog_use}.{schema_use}.advanced_churn_feature_table",
    label="Feature Table",
)

# Feature table to store the computed features.
dbutils.widgets.text(
    "avg_price_increase",
    f"{catalog_use}.{schema_use}.avg_price_increase",
    label="Avg Price Increase Function",
)


# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../features

# COMMAND ----------

input_table_name = dbutils.widgets.get("input_table_name")
advanced_churn_label_table = dbutils.widgets.get("advanced_churn_label_table")
advanced_churn_feature_table = dbutils.widgets.get("advanced_churn_feature_table")
avg_price_increase = dbutils.widgets.get("avg_price_increase")

assert input_table_name != "", "input_table_path notebook parameter must be specified"
assert advanced_churn_feature_table != "", "output_table_name notebook parameter must be specified"

# COMMAND ----------

print(f""" 
  input_table_name = {input_table_name}
  advanced_churn_label_table = {advanced_churn_label_table}
  advanced_churn_feature_table = {advanced_churn_feature_table}
  avg_price_increase = {avg_price_increase}
""")

# COMMAND ----------

raw_data = spark.table(input_table_name)
display(raw_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Featurization Logic(s) for BATCH feature computation - See the .py files located in the features folder
# MAGIC
# MAGIC 1. Compute the number of active services (compute_service_features.py which adds the column num_optional_services)
# MAGIC 2. Clean-up names and manual mapping (clean_churn_features.py)
# MAGIC
# MAGIC _This can also work for streaming based features_

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # Compute & Write to Feature Store
# MAGIC
# MAGIC Once our features are ready, we'll save them in a feature table. Any Delta Table registered to Unity Catalog can be a feature table.
# MAGIC
# MAGIC This will allow us to leverage Unity Catalog for governance, discoverability, and reusability of our features across our organization and increase team efficiency.
# MAGIC
# MAGIC The lineage capability in Unity Catalog brings traceability and governance to our deployment, knowing which model depends on which feature tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Features

# COMMAND ----------

# DBTITLE 1,Compute Churn Features and append a timestamp
# Compute the features. This is done by dynamically loading the features module.
# Add current scoring timestamp
from compute_service import compute_service_features
from featurization_function import clean_churn_features

this_time = (datetime.now()).timestamp()
churn_features_n_predsDF = clean_churn_features(compute_service_features(raw_data)) \
                            .withColumn("transaction_ts", lit(this_time).cast("timestamp"))

display(churn_features_n_predsDF)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract ground-truth labels in a separate table to avoid label leakage
# MAGIC * In reality, ground-truth label data should be in a separate table

# COMMAND ----------

# DBTITLE 1,Write a Labels table with customer_id, transaction_ts, churn, and split
# Best practice: specify train-val-test split as categorical label (to be used by automl and/or model validation jobs)
train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1

churn_features_n_predsDF.select("customer_id", "transaction_ts", "churn") \
                        .withColumn("random", F.rand(seed=42)) \
                        .withColumn("split",
                                    F.when(F.col("random") < train_ratio, "train")
                                    .when(F.col("random") < train_ratio + val_ratio, "validate")
                                    .otherwise("test")) \
                        .drop("random") \
                        .write.format("delta") \
                        .mode("overwrite").option("overwriteSchema", "true") \
                        .saveAsTable(advanced_churn_label_table)

churn_featuresDF = churn_features_n_predsDF.drop("churn")

# COMMAND ----------

# MAGIC %md
# MAGIC Add primary key constraints to the label table for feature lookup

# COMMAND ----------

spark.sql(f"ALTER TABLE {advanced_churn_label_table} ALTER COLUMN customer_id SET NOT NULL")
spark.sql(f"ALTER TABLE {advanced_churn_label_table} ALTER COLUMN transaction_ts SET NOT NULL")
spark.sql(f"ALTER TABLE {advanced_churn_label_table} ADD CONSTRAINT advanced_churn_label_table_pk PRIMARY KEY(customer_id, transaction_ts)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to Feature Store

# COMMAND ----------

# MAGIC %md
# MAGIC ### The FS table is a centralized repository for managing, sharing, and serving ML features
# MAGIC * **Consistency between training and inference:** FS ensures consistent feature definitions and transformations between training and inference 
# MAGIC * **Lineage and Governance:** FS Knows which models, jobs and endpoints use each feature
# MAGIC * **Feature Reusability & Discovery:** Enables feature reuse across tealms, reducing duplication and computational costs
# MAGIC * **Integration with ML Workflows** FS are designed to work seamlessly with model training, scoring, and deployment pipelines, automatically retrieving the right features for your models

# COMMAND ----------

# DBTITLE 1,Drop the feature table if it already exists
spark.sql(f"DROP TABLE IF EXISTS {advanced_churn_feature_table}")

# COMMAND ----------

# DBTITLE 1,Import Feature Store Client
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# COMMAND ----------

# DBTITLE 1,Create "feature"/UC table
churn_feature_table = fe.create_table(
  name=advanced_churn_feature_table,
  primary_keys=["customer_id", "transaction_ts"],
  schema=churn_featuresDF.schema,
  timeseries_columns="transaction_ts",
  description=f"These features are derived from the {advanced_churn_feature_table} table in the lakehouse. We created service features and cleaned up their names.  No aggregations were performed. [Warning: This table doesn't store the ground truth and can now be used with AutoML's feature table integration."
)

# COMMAND ----------

# DBTITLE 1,Nothings here because we haven't written it yet, we just created the table layout
display(spark.table(f"{advanced_churn_feature_table}"))

# COMMAND ----------

# DBTITLE 1,Write the feature values to a feature table
fe.write_table(
  name=advanced_churn_feature_table,
  df=churn_featuresDF, # can be a streaming dataframe as well
  mode='merge' #'merge' supports schema evolution
)

# COMMAND ----------

# DBTITLE 1,We have all the features without Churn
display(spark.table(f"{advanced_churn_feature_table}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define Featurization Logic for on-demand feature functions
# MAGIC
# MAGIC We will define a function for features that need to be calculated on demand. These functions can be used in batch/offline and serving/online inference.
# MAGIC
# MAGIC It is common that customers who have elevated monthly bills have a higher propensity to churn. The `avg_price_increase` function calculates the potential average price increase based on their historical charges and current tenure. The function lets the model use this freshly calculated value as a feature for training and, later, scoring.
# MAGIC
# MAGIC This function is defined under Unity Catalog, which provides governance over who can use the function.
# MAGIC
# MAGIC Please take a look at the documentation for more information. ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/on-demand-features.html)|[Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/feature-store/on-demand-features)) 

# COMMAND ----------

output_schema = advanced_churn_feature_table.split(".")[0]
output_database = advanced_churn_feature_table.split(".")[1]
spark.sql(f"USE CATALOG {output_schema}");
spark.sql(f"USE SCHEMA {output_database}")

# COMMAND ----------

# MAGIC %sql
# MAGIC   CREATE OR REPLACE FUNCTION avg_price_increase(monthly_charges_in DOUBLE, tenure_in DOUBLE, total_charges_in DOUBLE)
# MAGIC   RETURNS FLOAT
# MAGIC   LANGUAGE PYTHON
# MAGIC   COMMENT "[Feature Function] Calculate potential average price increase for tenured customers based on last monthly charges and updated tenure"
# MAGIC   AS $$
# MAGIC   if tenure_in > 0:
# MAGIC     return monthly_charges_in - total_charges_in/tenure_in
# MAGIC   else:
# MAGIC     return 0
# MAGIC   $$

# COMMAND ----------

dbutils.notebook.exit(0)
