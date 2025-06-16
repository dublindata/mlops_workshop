# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction Model Inference
# MAGIC
# MAGIC ## Inference with the Champion model
# MAGIC
# MAGIC Models in Unity Catalog can be loaded for use in batch inference pipelines. Generated predictions would be used to advise on customer retention strategies or be used for analytics. The model in use is the __@Champion__ model, and we will load it for use in our pipeline.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-5.png?raw=true" width="1200">
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=984752964297111&notebook=%2F02-mlops-advanced%2F05_batch_inference&demo_name=mlops-end2end&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fmlops-end2end%2F02-mlops-advanced%2F05_batch_inference&version=1&user_hash=a3692eff9e5299c6a85c26f2dc27b2e2000517102cea778a7cc80efff9afb355">

# COMMAND ----------

# DBTITLE 1,Install MLflow version for model lineage in UC [for MLR < 15.2]
# MAGIC %pip install --quiet mlflow==2.19 databricks-feature-engineering==0.8.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Feature table to store the computed features.
dbutils.widgets.text(
    "inference_df",
    "dev.koeppen_dabs_demo.advanced_churn_inference_table",
    label="Inference Table",
)

# Feature table to store the computed features.
dbutils.widgets.text(
    "model_name",
    "dev.koeppen_dabs_demo.advanced_mlops_churn_model",
    label="experiment_name",
)

# Feature table to store the computed features.
dbutils.widgets.text(
    "advanced_churn_inference_results",
    "dev.koeppen_dabs_demo.advanced_churn_inference_results",
    label="Inference Results Table",
)

# COMMAND ----------

inference_df=dbutils.widgets.get("inference_df")
model_name=dbutils.widgets.get("model_name")
advanced_churn_inference_results=dbutils.widgets.get("advanced_churn_inference_results")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run inferences

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch inference on the Champion model
# MAGIC
# MAGIC We are ready to run inference on the Champion model. We will leverage the feature engineering client's `score_batch` method and generate predictions for our customer records.
# MAGIC
# MAGIC For simplicity, we assume that features have been pre-computed for all new customer records and already stored in a feature table. These are typically done by separate feature engineering pipelines.

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
champion_model_details = client.get_model_version_by_alias(model_name, "Champion")
champion_model_version = int(champion_model_details.version)
champion_run_info = client.get_run(run_id=champion_model_details.run_id)

print(f"Champion model for {model_name} is model version {champion_model_version}")

# COMMAND ----------

# DBTITLE 1,In a python notebook
# Batch score
preds_df = fe.score_batch(df=inference_df, model_uri=model_uri, result_type="string")
display(preds_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC That's it! Our data can now be saved as a table and reused by the Data Analyst / Marketing team to take special action and reduce Churn risk on these customers!
# MAGIC
# MAGIC Your data will also be available within Genie to answer any churn-related question using plain text English!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save the predictions for monitoring
# MAGIC
# MAGIC Since we want to monitor the model and its predictions over time, we will save the predictions, along with information on the model used to produce them. This table can then be monitored for feature drift and prediction drift.
# MAGIC
# MAGIC Note that this table does not have the ground truth labels. These are usually collected and made available over time, and in many cases, may not even be available! However, this does not stop us from monitoring the data for drift, as that alone may be a sign that the model has to be retrained.
# MAGIC
# MAGIC The table displayed below is saved into `advanced_churn_offline_inference`. It includes the model version used for scoring, the model alias, the predictions, and the timestamp when the inference was made. It does not contain any labels.
# MAGIC

# COMMAND ----------

from mlflow import MlflowClient
from datetime import datetime
client = MlflowClient()

model = client.get_registered_model(name=model_name)
model_version = int(client.get_model_version_by_alias(name=model_name, alias="Champion").version)

# COMMAND ----------

import pyspark.sql.functions as F
from datetime import datetime, timedelta

offline_inference_df = preds_df.withColumn("model_name", F.lit(model_name)) \
                              .withColumn("model_version", F.lit(model_version)) \
                              .withColumn("model_alias", F.lit("Champion")) \
                              .withColumn("inference_timestamp", F.lit(datetime.now()- timedelta(days=2)))

offline_inference_df.write.mode("append") \
                    .saveAsTable(advanced_churn_inference_results)

display(offline_inference_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Congratulations! You have successfully used the model for batch inference.
# MAGIC
# MAGIC Let's look at how we can deploy this model as a REST API endpoint for real-time inference.
# MAGIC
# MAGIC Next:  [Serve the features and model in real-time]($./06_serve_features_and_model)
