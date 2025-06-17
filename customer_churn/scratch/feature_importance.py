# Databricks notebook source
# MAGIC %md
# MAGIC # Use the best AutoML generated notebook to bootstrap our ML Project
# MAGIC
# MAGIC We have selected the notebook from the best run of the AutoML experiment and reused it to build our model.
# MAGIC
# MAGIC AutoML generates the code in this notebook automatically. As a Data Scientist, I can tune it based on my business knowledge if needed.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/mlops/advanced/banners/mlflow-uc-end-to-end-advanced-2.png?raw=True" width="1200">
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=984752964297111&notebook=%2F02-mlops-advanced%2F02_automl_champion&demo_name=mlops-end2end&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fmlops-end2end%2F02-mlops-advanced%2F02_automl_champion&version=1&user_hash=a3692eff9e5299c6a85c26f2dc27b2e2000517102cea778a7cc80efff9afb355">
# MAGIC <!-- [metadata={"description":"MLOps end2end workflow: Auto-ML notebook",
# MAGIC  "authors":["quentin.ambard@databricks.com"],
# MAGIC  "db_resources":{},
# MAGIC   "search_tags":{"vertical": "retail", "step": "Data Engineering", "components": ["auto-ml"]},
# MAGIC                  "canonicalUrl": {"AWS": "", "Azure": "", "GCP": ""}}] -->

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC
# MAGIC

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk==0.23.0 mlflow==2.19 databricks-feature-engineering==0.8.0 databricks-automl-runtime==0.2.21 holidays==0.64 category_encoders==2.7.0 hyperopt==0.2.7 shap==0.46.0 lightgbm==4.5.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.

# Notebook Environment
dbutils.widgets.dropdown("env", "dev", ["dev","staging", "prod"], "Environment Name")
env = dbutils.widgets.get("env")

# MLflow experiment name.
dbutils.widgets.text(
    "experiment_name",
    f"/advanced_mlops_churn_experiment",
    label="MLflow experiment name",
)

# Unity Catalog registered model name to use for the trained mode.
dbutils.widgets.text(
    "model_name", "dev.koeppen_dabs_demo.advanced_mlops_churn_model", label="Full (Three-Level) Model Name"
)


# Feature table to store the computed features.
dbutils.widgets.text(
    "advanced_churn_label_table",
    "dev.koeppen_dabs_demo.advanced_churn_label_table",
    label="advanced_churn_label_table",
)

# Feature table to store the computed features.
dbutils.widgets.text(
    "advanced_churn_feature_table",
    "dev.koeppen_dabs_demo.advanced_churn_feature_table",
    label="advanced_churn_feature_table",
)

# Feature table to store the computed features.
dbutils.widgets.text(
    "avg_price_increase",
    "dev.koeppen_dabs_demo.avg_price_increase",
    label="avg_price_increase",
)



# COMMAND ----------

experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
avg_price_increase=dbutils.widgets.get("avg_price_increase")
advanced_churn_label_table=dbutils.widgets.get("advanced_churn_label_table")
advanced_churn_feature_table=dbutils.widgets.get("advanced_churn_feature_table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature importance
# MAGIC
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, so we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = True

# COMMAND ----------

if shap_enabled:
    mlflow.autolog(disable=True)
    mlflow.sklearn.autolog(disable=True)
    from shap import KernelExplainer, summary_plot
    # SHAP cannot explain models using data with nulls.
    # To enable SHAP to succeed, both the background data and examples to explain are imputed with the mode (most frequent values).
    mode = X_train.mode().iloc[0]

    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, X_train.shape[0]), random_state=790671489).fillna(mode)

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(100, X_val.shape[0]), random_state=790671489).fillna(mode)

    # Use Kernel SHAP to explain feature importance on the sampled rows from the validation set.
    predict = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="logit")
    shap_values = explainer.shap_values(example, l1_reg=False, nsamples=100)
    summary_plot(shap_values, example, class_names=model.classes_)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Confusion matrix, ROC, and Precision-Recall curves for validation data
# MAGIC
# MAGIC We show the confusion matrix, RO,C and Precision-Recall curves of the model on the validation data.
# MAGIC
# MAGIC For the plots evaluated on the training and the test data, check the artifacts on the MLflow run page.

# COMMAND ----------

# Click the link to see the MLflow run page
displayHTML(f"<a href=#mlflow/experiments/{mlflow_run.info.experiment_id}/runs/{ mlflow_run.info.run_id }/artifactPath/model> Link to model run page </a>")

# COMMAND ----------

import uuid
from IPython.display import Image
import os
# Create temp directory to download MLflow model artifact
eval_temp_dir = os.path.join(os.environ["SPARK_LOCAL_DIRS"], "tmp", str(uuid.uuid4())[:8])
os.makedirs(eval_temp_dir, exist_ok=True)

# Download the artifact
eval_path = mlflow.artifacts.download_artifacts(run_id=mlflow_run.info.run_id, dst_path=eval_temp_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confusion matrix for validation dataset

# COMMAND ----------

eval_confusion_matrix_path = os.path.join(eval_path, "val_confusion_matrix.png")
display(Image(filename=eval_confusion_matrix_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### ROC curve for validation dataset

# COMMAND ----------

eval_roc_curve_path = os.path.join(eval_path, "val_roc_curve_plot.png")
display(Image(filename=eval_roc_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision-Recall curve for validation dataset

# COMMAND ----------

eval_pr_curve_path = os.path.join(eval_path, "val_precision_recall_curve_plot.png")
display(Image(filename=eval_pr_curve_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Automate model promotion validation
# MAGIC
# MAGIC Next step: [Search runs and trigger model promotion validation]($./03_from_notebook_to_models_in_uc)

# COMMAND ----------


