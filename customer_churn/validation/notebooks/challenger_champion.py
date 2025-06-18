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

# MAGIC %md
# MAGIC ## Fetch Model information
# MAGIC
# MAGIC We will fetch the model information for the __Challenger__ model from Unity Catalog.

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

# Feature table to store the computed features.
dbutils.widgets.text(
    "model_name",
    f"{catalog_use}.{schema_use}.advanced_mlops_churn_model",
    label="Model Name",
)

# Feature table to store the computed features.
dbutils.widgets.text(
    "model_info_table",
    f"{catalog_use}.{schema_use}.model_info_table",
    label="model_info_table",
)

# COMMAND ----------

model_info_table = dbutils.widgets.get("model_info_table")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

print(f""" 
  model_info_table: {model_info_table}
  model_name: {model_name}
""")

# COMMAND ----------

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import pyspark.sql.functions as F
from pyspark.sql import Window
import datetime

# Setup
client = MlflowClient()

# Step 1: Get current aliased versions from registry
try:
    champion_version = client.get_model_version_by_alias(model_name, "champion").version
    challenger_version = client.get_model_version_by_alias(model_name, "challenger").version
except MlflowException as e:
    if "champion" in str(e):
        print("No Champion")
    elif "challenger" in str(e):
        print("No Challenger")
    exit()

# COMMAND ----------

# Step 2: Read model info table
df = spark.read.table(model_info_table).filter(F.col("model_name") == model_name)

# Step 3: Filter to just the two aliased versions
df_filtered = df.filter(F.col("model_version").isin(int(champion_version), int(challenger_version)))

# Step 4: Use a window to get latest row per version
window_spec = Window.partitionBy("model_version").orderBy(F.desc("validation_timestamp"))
df_latest = df_filtered.withColumn("row_num", F.row_number().over(window_spec)).filter("row_num = 1")

# Step 5: Collect the two latest records
records = df_latest.collect()
if len(records) < 2:
    raise ValueError("Missing inference results for either the champion or challenger model version.")

# Assign based on F1 score
model_1 = records[0]
model_2 = records[1]

if model_1.f1_score >= model_2.f1_score:
    new_champion, new_challenger = model_1, model_2
else:
    new_champion, new_challenger = model_2, model_1

# Step 6: Update aliases
# First remove old aliases (best practice)
client.delete_registered_model_alias(model_name, "champion")
client.delete_registered_model_alias(model_name, "challenger")

# Demote the previous champion (if it has changed)
if str(new_champion.model_version) != str(champion_version):
    client.set_registered_model_alias(model_name, version=champion_version, alias="prior_champion")
    client.set_model_version_tag(model_name, champion_version, "status", "demoted_to_prior_champion")
    client.set_model_version_tag(model_name, champion_version, "demoted_on", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    client.set_model_version_tag(model_name, challenger_version, "status", "promoted_to_champion")
    client.set_model_version_tag(model_name, challenger_version, "promoted_on", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Demoted v{champion_version} to prior_champion")
    dbutils.jobs.taskValues.set("new_champion", "true")
else:
    dbutils.jobs.taskValues.set("new_champion", "false")


# Reassign
client.set_registered_model_alias(model_name, version=new_champion.model_version, alias="champion")
client.set_registered_model_alias(model_name, version=new_challenger.model_version, alias="challenger")

# We can also tag the model version with the F1 score for visibility
print(f"Champion: v{new_champion.model_version} (F1 = {new_champion.f1_score})")
print(f"Challenger: v{new_challenger.model_version} (F1 = {new_challenger.f1_score})")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Congratulations, our model is now validated and promoted accordingly
# MAGIC
# MAGIC We now know that our model is ready to be used in inference pipelines and real-time serving endpoints, as it matches our validation standards.
