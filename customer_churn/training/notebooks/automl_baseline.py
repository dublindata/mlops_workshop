# Databricks notebook source
# MAGIC %md
# MAGIC # Run AutoML and register the best model

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## Accelerating Churn model creation using Databricks Auto-ML
# MAGIC ### A glass-box solution that empowers data teams without taking away control
# MAGIC
# MAGIC Databricks simplifies model creation and MLOps. However, bootstrapping new ML projects can still be long and inefficient.
# MAGIC
# MAGIC Instead of creating the same boilerplate for each new project, Databricks Auto-ML can automatically generate state-of-the-art models for Classifications, regression, and forecasts.
# MAGIC
# MAGIC Models can be directly deployed or leverage generated notebooks to bootstrap projects with best practices, saving you weeks of effort.
# MAGIC
# MAGIC <img width="1000" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/auto-ml-full.png"/>
# MAGIC
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC ### Using Databricks Auto ML with our Churn dataset
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img style="float: right" width="600" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn-auto-ml.png"/>
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC Auto ML is available under **Machine Learning - Experiments**. All we have to do is create a new AutoML experiment, select the table containing the ground-truth labels, and join it with the features in the feature table.
# MAGIC
# MAGIC Our prediction target is the `churn` column.
# MAGIC
# MAGIC Click on **Start**, and Databricks will do the rest.
# MAGIC
# MAGIC While this is done using the UI, you can also leverage the [Python API](https://docs.databricks.com/applications/machine-learning/automl.html#automl-python-api-1)
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC #### Join/Use features directly from the Feature Store from the [UI](https://docs.databricks.com/machine-learning/automl/train-ml-model-automl-ui.html#use-existing-feature-tables-from-databricks-feature-store) or [python API]()
# MAGIC * Select the table containing the ground-truth labels (i.e., `dbdemos.schema.churn_label_table`)
# MAGIC * Join remaining features from the feature table (i.e., `dbdemos.schema.churn_feature_table`)
# MAGIC
# MAGIC Please take a look at the __Quickstart__ version of this demo for an example of AutoML in action.

# COMMAND ----------

# MAGIC %pip install --quiet mlflow==2.19 databricks-feature-engineering==0.8.0

# COMMAND ----------

# DBTITLE 1,if there are any updates to our .py files it'll reflect
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path
%cd ../features

# COMMAND ----------

# MAGIC %md
# MAGIC # Define Variables

# COMMAND ----------

dbutils.widgets.text("catalog_use", "datascience_dev", label="Catalog to Use")
dbutils.widgets.text("schema_use", "main", label="Schema to Use")
dbutils.widgets.text("model_timeout_minutes", "5", label="Model Timeout Minutes")

# COMMAND ----------

catalog_use = dbutils.widgets.get("catalog_use")
schema_use = dbutils.widgets.get("schema_use")
spark.sql(f"USE {catalog_use}.{schema_use}")

# COMMAND ----------

# MAGIC %sql
# MAGIC select current_catalog(), current_schema();

# COMMAND ----------

# dbutils.widgets.removeAll()

# COMMAND ----------

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

# Feature table to store the computed features.
dbutils.widgets.text(
    "experiment_name",
    f"{catalog_use}.{schema_use}.advanced_mlops_churn_experiment",
    label="Experiment Name",
)

# Feature table to store the computed features.
dbutils.widgets.text(
    "model_name",
    f"{catalog_use}.{schema_use}.advanced_mlops_churn_model",
    label="Model Name",
)

# # Feature table to store the computed features.
# dbutils.widgets.text(
#     "features_from_registered_automl_model",
#     "dev.koeppen_dabs_demo.features_from_registered_automl_model",
#     label="features_from_registered_automl_model",
# )

# COMMAND ----------

model_timeout_minutes = int(dbutils.widgets.get("model_timeout_minutes"))
advanced_churn_label_table = dbutils.widgets.get("advanced_churn_label_table")
advanced_churn_feature_table = dbutils.widgets.get("advanced_churn_feature_table")
avg_price_increase = dbutils.widgets.get("avg_price_increase")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
# features_from_registered_automl_model = dbutils.widgets.get("features_from_registered_automl_model")

# COMMAND ----------

print(f""" 
  model_timeout_minutes: {model_timeout_minutes}
  advanced_churn_label_table: {advanced_churn_label_table}
  advanced_churn_feature_table: {advanced_churn_feature_table}
  avg_price_increase: {avg_price_increase}
  experiment_name: {experiment_name}
  model_name: {model_name}
""")

# COMMAND ----------

# output_schema = advanced_churn_feature_table.split(".")[0]
# output_database = advanced_churn_feature_table.split(".")[1]
# spark.sql(f"USE CATALOG {output_schema}");
# spark.sql(f"USE SCHEMA {output_database}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### labels_df has our customer_id, transaction_ts, churn, and split values

# COMMAND ----------

labels_df = spark.table(advanced_churn_label_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Our advanced_churn_feature_table has all the features we extracted from the CSV and pre-processing we did in Data Preparation (it doesn't have churn label)

# COMMAND ----------

# DBTITLE 1,joins raw data with features
from databricks.feature_store import FeatureFunction, FeatureLookup

feature_lookups = [
    FeatureLookup(
      table_name=advanced_churn_feature_table,
      lookup_key=["customer_id"],
      timestamp_lookup_key="transaction_ts"
    ),
    FeatureFunction(
      udf_name=avg_price_increase,
      input_bindings={
        "monthly_charges_in" : "monthly_charges",
        "tenure_in" : "tenure",
        "total_charges_in" : "total_charges"
      },
      output_name="avg_price_increase"
    )
]

# Step 1: Read features
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# Create Feature specifications object
training_set_specs = fe.create_training_set(
  df=labels_df, # DataFrame with lookup keys and label/target (+ any other input)
  label="churn",
  feature_lookups=feature_lookups,
  exclude_columns=["customer_id", "transaction_ts", 'split']
)
training_df = training_set_specs.load_df()

# COMMAND ----------

display(training_df)

# COMMAND ----------

# DBTITLE 1,Imports and Utility Functions
import mlflow
import databricks.automl
from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F

def start_automl_run(dataset, target_col, experiment_name=None, timeout_minutes=15):
    return databricks.automl.classify(
        dataset=dataset,
        target_col=target_col,
        timeout_minutes=timeout_minutes,
        experiment_name=experiment_name
    )


# COMMAND ----------

# DBTITLE 1,Run the experiment
automl_result = start_automl_run(
    dataset=training_df,         
    target_col="churn",
    timeout_minutes=model_timeout_minutes,
    experiment_name=experiment_name
)
best_model_uri = automl_result.best_trial.model_path
best_run_id= automl_result.best_trial.mlflow_run_id
print(f"Best model run ID: {best_run_id}")
print(f"Registered champion model: {best_model_uri}")


# COMMAND ----------

# DBTITLE 1,Register the best model in UC
from mlflow import register_model

registration = mlflow.register_model(
    model_uri=best_model_uri,
    name=model_name
)

print("Model version:", registration.version)
print("Run ID:", registration.run_id)
version=registration.version
run_id=registration.run_id

# COMMAND ----------

# DBTITLE 1,Register the model in UC's Model Registry
# MAGIC %md
# MAGIC from mlflow import register_model
# MAGIC from mlflow.tracking import MlflowClient
# MAGIC
# MAGIC client = MlflowClient()
# MAGIC
# MAGIC fe = FeatureEngineeringClient()
# MAGIC fe.log_model(
# MAGIC     model=best_model_uri,
# MAGIC     artifact_path="automl_model",
# MAGIC     flavor=mlflow.pyfunc,
# MAGIC     training_set=training_df,
# MAGIC     name=model_name,
# MAGIC     input_example=training_df.limit(5).toPandas(),
# MAGIC     description="AutoML model with feature lineage"
# MAGIC )
# MAGIC
# MAGIC
# MAGIC
# MAGIC versions = client.search_model_versions(f"run_id='{best_run_id}' and name='{model_name}'")
# MAGIC model_version_details = client.get_model_version(name=model_name, version=versions)
# MAGIC
# MAGIC run_id=model_version_details.run_id
# MAGIC
# MAGIC print("Model version:", versions)
# MAGIC print("Run ID:", run_id)
# MAGIC

# COMMAND ----------

# DBTITLE 1,Assigns Champion Alias
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Assign alias
if registration.version == '1':
    client.set_registered_model_alias(name=model_name, 
                                      alias="champion", 
                                      version=registration.version)
else:
    client.set_registered_model_alias(name=model_name, 
                                      alias="challenger", 
                                      version=registration.version)

print(f"Assigned alias {'champion' if registration.version == '1' else 'challenger'} to version {registration.version}")

# COMMAND ----------

# DBTITLE 1,Get what model AutoML decided was the best
import mlflow.sklearn
# Load the model
pipeline_model = mlflow.sklearn.load_model(best_model_uri)
# Get the last step (the actual estimator)
estimator = pipeline_model.steps[-1][1]  

# COMMAND ----------

# DBTITLE 1,Apply various tags to the model for documentation
client = MlflowClient()

# We can also tag the model version with the F1 score for visibility
client.set_model_version_tag(
  name=model_name,
  version=version,
  key="model_type",
  value=f"{type(estimator).__name__}"
)

# We can also tag the model version with the F1 score for visibility
client.set_model_version_tag(
  name=model_name,
  version=version,
  key="modeling_method",
  value="AutoML"
)

# We can also tag the model version with the F1 score for visibility
client.set_model_version_tag(
  name=model_name,
  version=version,
  key="best_run_id",
  value=best_run_id
)

# COMMAND ----------

dbutils.notebook.exit(0)
