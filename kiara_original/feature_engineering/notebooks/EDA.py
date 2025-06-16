# Databricks notebook source
# MAGIC %md
# MAGIC # Churn Prediction EDA
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

# MAGIC %md
# MAGIC # Define Variables

# COMMAND ----------

# A Hive-registered Delta table containing the input data.
dbutils.widgets.text(
    "input_table_name",
    "main.dbdemos_mlops.advanced_churn_bronze_customers",
    label="Input Table Name",
)

# COMMAND ----------

input_table_name = dbutils.widgets.get("input_table_name")

assert input_table_name != "", "input_table_path notebook parameter must be specified"

# COMMAND ----------

raw_data = spark.table(input_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis
# MAGIC To get a feel of the data, what needs cleaning, pre-processing, etc.
# MAGIC - **Use Databricks's native visualization tools**
# MAGIC - Bring your own visualization library of choice (i.e., seaborn, plotly)

# COMMAND ----------

# DBTITLE 1,Read in Bronze Delta table using Spark
# Read into a Spark dataframe
telcoDF = spark.read.table(input_table_name)
display(telcoDF)

# COMMAND ----------

telco_df = spark.read.table(input_table_name).pandas_api()
telco_df["internet_service"].value_counts().plot.pie()

# COMMAND ----------

dbutils.notebook.exit(0)
