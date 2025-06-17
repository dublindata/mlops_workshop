# Databricks notebook source
# MAGIC %md
# MAGIC # Challenger model validation
# MAGIC
# MAGIC This notebook performs validation tasks on the candidate __Challenger__ model.
# MAGIC
# MAGIC It goes through a few steps to validate the model before labelling it (by setting its alias) to `Challenger`.
# MAGIC
# MAGIC When organizations first start to put MLOps processes in place, they should consider having a "human-in-the-loop" to perform visual analyses to validate models before promoting them. As they get more familiar with the process, they can consider automating the steps in a __Workflow__ . The benefits of automation is to ensure that these validation checks are systematically performed before new models are integrated into inference pipelines or deployed for realtime serving. Of course, organizations can opt to retain a "human-in-the-loop" in any part of the process and put in place the degree of automation that suits its business needs.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-4.png?raw=true" width="1200">
# MAGIC
# MAGIC *Note: In a typical mlops setup, this would run as part of an automated job to validate a new model. We'll run this demo as an interactive notebook.*
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable the collection or the tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=984752964297111&notebook=%2F02-mlops-advanced%2F04_challenger_validation&demo_name=mlops-end2end&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fmlops-end2end%2F02-mlops-advanced%2F04_challenger_validation&version=1&user_hash=a3692eff9e5299c6a85c26f2dc27b2e2000517102cea778a7cc80efff9afb355">

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## General Validation Checks
# MAGIC
# MAGIC <!--img style="float: right" src="https://github.com/QuentinAmbard/databricks-demo/raw/main/retail/resources/images/churn-mlflow-webhook-1.png" width=600 -->
# MAGIC
# MAGIC In the context of MLOps, there are more tests than simply how accurate a model will be.  To ensure the stability of our ML system and compliance with any regulatory requirements, we will subject each model added to the registry to a series of validation checks.  These include, but are not limited to:
# MAGIC <br>
# MAGIC * __Model documentation__
# MAGIC * __Inference on production data__
# MAGIC * __Champion-Challenger testing to ensure that business KPIs are acceptable__
# MAGIC
# MAGIC In this notebook, we explore some approaches to performing these tests, and how we can add metadata to our models by tagging if they have passed a given test.
# MAGIC
# MAGIC This part is typically specific to your line of business and quality requirements.
# MAGIC
# MAGIC For each test, we'll add information using tags to know what has been validated in the model. We can also add Comments to a model if needed.

# COMMAND ----------

# MAGIC %pip install --quiet mlflow==2.19 databricks-feature-engineering==0.8.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Defining parameters we're going to set in YAML file
dbutils.widgets.text("catalog_use", "datascience_dev", label="Catalog to Use")
dbutils.widgets.text("schema_use", "main", label="Schema to Use")

# COMMAND ----------

catalog_use = dbutils.widgets.get("catalog_use")
schema_use = dbutils.widgets.get("schema_use")
spark.sql(f"USE {catalog_use}.{schema_use}")

# COMMAND ----------

# DBTITLE 1,make sure we're using the expected catalog/schema
# MAGIC %sql
# MAGIC select current_catalog(), current_schema();

# COMMAND ----------

# DBTITLE 1,Defining parameters we're going to set in YAML file
# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    "/advanced_mlops_churn_experiment",
    label="Experiment Name",
)

# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", 
    f"{catalog_use}.{schema_use}.advanced_mlops_churn_model", 
    label="Full (Three-Level) Model Name"
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

# Feature table to store the computed features.
dbutils.widgets.dropdown(
    "model_alias",
    "champion",
    ["challenger", "champion", "prior_champion"],
    label="Model Alias"
)

# Feature table to store the computed features.
dbutils.widgets.text(
    "model_info_table",
    f"{catalog_use}.{schema_use}.model_info_table",
    label="Model Information Table",
)

# COMMAND ----------

advanced_churn_label_table = dbutils.widgets.get("advanced_churn_label_table")
advanced_churn_feature_table = dbutils.widgets.get("advanced_churn_feature_table")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
avg_price_increase=dbutils.widgets.get("avg_price_increase")
model_alias=dbutils.widgets.get("model_alias")
model_info_table=dbutils.widgets.get("model_info_table")

# COMMAND ----------

# DBTITLE 1,Making sure we're using the parameters we're expecting
print(f""" 
  advanced_churn_label_table: {advanced_churn_label_table}
  advanced_churn_feature_table: {advanced_churn_feature_table}
  avg_price_increase: {avg_price_increase}
  experiment_name: {experiment_name}
  model_name: {model_name}
  model_alias: {model_alias}
  model_info_table: {model_info_table}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # Getting Model Information based on which Alias we're wanting to Validate

# COMMAND ----------

from mlflow.tracking import MlflowClient

# COMMAND ----------

client = MlflowClient()
model_details = client.get_model_version_by_alias(model_name, model_alias)
model_version = int(model_details.version)
model_uri = model_uri = f"models:/{model_name}/{model_version}"
# Determine modeling method via tag
modeling_method = model_details.tags.get("modeling_method", "")

print(f"Validating {model_alias} model for {model_name} on model version {model_version}")

# COMMAND ----------

# Load label table
labels_df = spark.table(advanced_churn_label_table)
# Load feature table
features_df = spark.table(advanced_churn_feature_table)


# COMMAND ----------

# DBTITLE 1,Batch scoring the validation DF to get the F1 Score
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient
from mlflow import pyfunc
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import Row
import pyspark.sql.functions as F

# Load model version via alias
client = MlflowClient()
fe = FeatureEngineeringClient()

# AutoML models are often exported as MLflow pyfunc models which is designed to accept input data as pandas DFs
if modeling_method == "AutoML":
    print("Using pyfunc for AutoML scoring...")

    model = pyfunc.load_model(model_uri)

    # Join labels + features
    features_df = features_df.withColumn("avg_price_increase",(F.col("monthly_charges") - (F.col("total_charges") / F.col("tenure"))))
    joined_df = labels_df.join(features_df, on=["customer_id", "transaction_ts"], how="inner")

    # Select only the columns the model expects
    input_schema = model.metadata.get_input_schema()
    expected_cols = [col.name for col in input_schema.inputs]

    validation_df = joined_df
    pdf_features = validation_df.select(*expected_cols).toPandas()
    # Convert all float64 columns to float32 to match model input schema
    pdf_features = pdf_features.astype({col: "float32" for col in pdf_features.select_dtypes("float64").columns})

    pdf_labels = validation_df.select("churn").toPandas()

    predictions = model.predict(pdf_features)

    from sklearn.metrics import f1_score
    f1 = f1_score(pdf_labels, predictions, pos_label="Yes")

    print(f"F1 Score: {f1:.4f}")

else:
    print("Using Feature Store batch scoring...")

    model_uri_with_alias = f"models:/{model_name}@{model_alias}"

    # Ensure predictions are numeric
    scored_df = fe.score_batch(df=labels_df, model_uri=model_uri_with_alias, result_type="string")

    # Convert label + prediction to DoubleType for evaluator
    scored_df = (
        scored_df
        .withColumn("prediction", F.when(F.col("prediction") == "Yes", 1.0).otherwise(0.0))
        .withColumn("churn", F.when(F.col("churn") == "Yes", 1.0).otherwise(0.0))
    )

    evaluator = MulticlassClassificationEvaluator(labelCol="churn", predictionCol="prediction", metricName="f1")
    f1 = evaluator.evaluate(scored_df)

    print(f"F1 Score: {f1:.4f}")

# COMMAND ----------

from pyspark.sql import SparkSession
from datetime import datetime

spark = SparkSession.builder.getOrCreate()
df_spark = spark.createDataFrame([(
  model_name,
  model_version,
  float(round(f1, 4)),
  model_alias, 
  modeling_method,
  datetime.now()
)], ["model_name", "model_version", "f1_score","model_alias","modeling_method","validation_timestamp"]).write.mode("append").saveAsTable(model_info_table)


# COMMAND ----------


