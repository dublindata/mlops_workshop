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

# DBTITLE 1,if there are any updates to our .py files it'll reflect
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

import os
notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

# COMMAND ----------

# MAGIC %pip install --quiet databricks-sdk==0.23.0 mlflow==2.19 databricks-feature-engineering==0.8.0 databricks-automl-runtime==0.2.21 holidays==0.64 category_encoders==2.7.0 hyperopt==0.2.7 shap==0.46.0 lightgbm==4.5.0

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Defining parameters we're going to set in YAML file
dbutils.widgets.text("catalog_use", "datascience_dev", label="Catalog to Use")
dbutils.widgets.text("schema_use", "main", label="Schema to Use")

# COMMAND ----------

# DBTITLE 1,Setting the Catalog and Schema so we know where to work out of
catalog_use = dbutils.widgets.get("catalog_use")
schema_use = dbutils.widgets.get("schema_use")
spark.sql(f"USE {catalog_use}.{schema_use}")

# COMMAND ----------

# DBTITLE 1,Make sure we're using the expected catalog & schema
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

# COMMAND ----------

experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
avg_price_increase=dbutils.widgets.get("avg_price_increase")
advanced_churn_label_table=dbutils.widgets.get("advanced_churn_label_table")
advanced_churn_feature_table=dbutils.widgets.get("advanced_churn_feature_table")
label_col="churn"

# COMMAND ----------

print(f""" 
  advanced_churn_label_table: {advanced_churn_label_table}
  advanced_churn_feature_table: {advanced_churn_feature_table}
  avg_price_increase: {avg_price_increase}
  experiment_name: {experiment_name}
  model_name: {model_name}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC # LightGBM Classifier training
# MAGIC - This is an auto-generated notebook.
# MAGIC - To reproduce these results, attach this notebook to a cluster with runtime version **15.4.x-cpu-ml-scala2.12**, and rerun it.
# MAGIC - Compare trials in the MLflow experiment
# MAGIC - Clone this notebook into your project folder by selecting **File > Clone** in the notebook toolbar.

# COMMAND ----------

# DBTITLE 1,Set the experiment & make sure model will be registered in UC
import mlflow
mlflow.set_experiment(experiment_name)
mlflow.set_registry_uri('databricks-uc') 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data
# MAGIC Load data directly from feature store
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We'll also use specific feature functions for on-demand features.
# MAGIC
# MAGIC Recall that we have defined the `avg_price_increase` feature function

# COMMAND ----------

# MAGIC %md
# MAGIC Create feature specifications.
# MAGIC
# MAGIC The feature lookup definition specifies the tables to use as feature tables and the key to lookup feature values.
# MAGIC
# MAGIC The feature function definition specifies which columns from the feature table are bound to the function inputs.
# MAGIC
# MAGIC The Feature Engineering client will use these values to create a training specification that's used to assemble the training dataset from the labels table and the feature table.

# COMMAND ----------

# DBTITLE 1,Define feature lookups
from databricks.feature_store import FeatureFunction, FeatureLookup

features = [
    FeatureLookup(
      table_name= advanced_churn_feature_table,
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

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Read the label table.

# COMMAND ----------

# DBTITLE 1,Pull labels to use for training/validating/testing
labels_df = spark.table(advanced_churn_label_table)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Create the training set specifications. This contains information on how the training set should be assembled from the label table, feature table, and feature function.

# COMMAND ----------

# DBTITLE 1,Create the training dataset
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# Create Feature specifications *object*
training_set_specs = fe.create_training_set(
  df=labels_df, # DataFrame with lookup keys and label/target (+ any other input)
  label="churn",
  feature_lookups=features,
  exclude_columns=["customer_id", "transaction_ts"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC With the training set specification, we can now build the training dataset.
# MAGIC
# MAGIC `training_set_specs.load_df()` returns a pySpark dataframe. We will convert it to a Pandas dataframe to train an LGBM model.

# COMMAND ----------

# DBTITLE 1,Load training set as Pandas dataframe
# Materialize the object above with .load_df() to create a Spark DF
# Convert the Spark DF to pandas so we can use LightGBM
df_loaded = training_set_specs.load_df().toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write training code
# MAGIC Once we have the dataset in a pandas DF
# MAGIC
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset with extra columns not used in training.
# MAGIC `[]` are dropped in the pipelines. Please take a look at the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
    
supported_cols = ["online_backup", "internet_service", "payment_method", "multiple_lines", "paperless_billing", "partner", "tech_support", "tenure", "contract", "avg_price_increase", "phone_service", "streaming_movies", "dependents", "senior_citizen", "num_optional_services", "device_protection", "monthly_charges", "total_charges", "streaming_tv", "gender", "online_security"]

col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train - Validation - Test Split
# MAGIC Split the training data into 3 sets based on the labels in Data Preparation:
# MAGIC - Train (70% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (10% of the dataset used to report the true performance of the model on an unseen dataset)
# MAGIC

# COMMAND ----------

# Now use the 'split' column instead of doing train_test_split
train_df = df_loaded[df_loaded["split"] == "train"]
val_df   = df_loaded[df_loaded["split"] == "validate"]
test_df  = df_loaded[df_loaded["split"] == "test"]

X_train = train_df.drop(["churn", "split"], axis=1)
y_train = train_df["churn"]

X_val = val_df.drop(["churn", "split"], axis=1)
y_val = val_df["churn"]

X_test = test_df.drop(["churn", "split"], axis=1)
y_test = test_df["churn"]


# COMMAND ----------

# MAGIC %md
# MAGIC ## Train classification model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under this MLflow experiment
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

import lightgbm
from lightgbm import LGBMClassifier

help(LGBMClassifier)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the objective function
# MAGIC The objective function used to find optimal hyperparameters. By default, this notebook only runs
# MAGIC this function once (`max_evals=1` in the `hyperopt.fmin` invocation) with fixed hyperparameters, but
# MAGIC hyperparameters can be tuned by modifying `space`, defined below. `hyperopt.fmin` will then use this
# MAGIC function's return value to search the space to minimize the loss.

# COMMAND ----------

from mlflow.models.signature import infer_signature
infer_signature(X_train,y_train)

# COMMAND ----------

import sys
sys.path.append("../../feature_engineering/features")  # Relative to current working directory
from mlflow.models import Model, infer_signature, ModelSignature
from mlflow.pyfunc import PyFuncModel
from mlflow import pyfunc
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline
from hyperopt import hp, tpe, fmin, STATUS_OK, Trials
import pandas as pd
from feature_engineering import get_bool_pipeline, get_numerical_pipeline, get_onehot_pipeline, get_preprocessor


# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
mlflow.sklearn.autolog(disable=True)
pipeline_val = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", get_preprocessor()),
])
pipeline_val.fit(X_train, y_train)
X_val_processed = pipeline_val.transform(X_val)

def objective(params):
  params["max_bin"] = int(params["max_bin"])
  params["max_depth"] = int(params["max_depth"])
  params["min_child_samples"] = int(params["min_child_samples"])
  params["n_estimators"] = int(params["n_estimators"])
  params["verbosity"] = -1
  params["num_leaves"] = int(params["num_leaves"])
  with mlflow.start_run() as mlflow_run: 
    lgbmc_classifier = LGBMClassifier(**params)

    model = Pipeline([
        ("column_selector", col_selector),
        ("preprocessor", get_preprocessor()),
        ("classifier", lgbmc_classifier),
    ])

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_models=False,
        silent=True)

    model.fit(X_train, y_train, classifier__callbacks=[lightgbm.early_stopping(5), lightgbm.log_evaluation(0)], classifier__eval_set=[(X_val_processed,y_val)])

    import warnings
    from mlflow.types.utils import _infer_schema
    from mlflow.exceptions import MlflowException

    # Log the model

    # Infer output schema
    try:
      output_schema = _infer_schema(y_train)
    except Exception as e:
      warnings.warn(f"Could not infer model output schema: {e}")
      output_schema = None
    
    # Use the Feature Engineering client to log the model
    # This logs the feature specifications along with the model,
    # allowing it to be used at inference time to retrieve features
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.sklearn,
        training_set=training_set_specs,
        output_schema=output_schema,
    )

    # Log metrics for the training set
    mlflow_model = Model()
    pyfunc.add_to_model(mlflow_model, loader_module="mlflow.sklearn")
    pyfunc_model = PyFuncModel(model_meta=mlflow_model, model_impl=model)
    training_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_train.assign(**{str(label_col):y_train}),
        targets=label_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "training_" , "pos_label": "Yes" }
    )
    lgbmc_training_metrics = training_eval_result.metrics

    # Log metrics for the validation set
    val_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_val.assign(**{str(label_col):y_val}),
        targets=label_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "val_" , "pos_label": "Yes" }
    )
    lgbmc_val_metrics = val_eval_result.metrics

    # Log metrics for the test set
    test_eval_result = mlflow.evaluate(
        model=pyfunc_model,
        data=X_test.assign(**{str(label_col):y_test}),
        targets=label_col,
        model_type="classifier",
        evaluator_config = {"log_model_explainability": False,
                            "metric_prefix": "test_" , "pos_label": "Yes" }
    )
    lgbmc_test_metrics = test_eval_result.metrics

    loss = -lgbmc_val_metrics["val_f1_score"]

    # Truncate metric key names so they can be displayed together
    lgbmc_val_metrics = {k.replace("val_", ""): v for k, v in lgbmc_val_metrics.items()}
    lgbmc_test_metrics = {k.replace("test_", ""): v for k, v in lgbmc_test_metrics.items()}

    return {
      "loss": loss,
      "status": STATUS_OK,
      "val_metrics": lgbmc_val_metrics,
      "test_metrics": lgbmc_test_metrics,
      "model": model,
      "run": mlflow_run,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure the hyperparameter search space
# MAGIC Configure the search space of parameters. Parameters below are all constant expressions, but can be
# MAGIC modified to widen the search space. For example, when training a decision tree classifier, to allow
# MAGIC the maximum tree depth to be either 2 or 3, set the key of 'max_depth' to
# MAGIC `hp.choice('max_depth', [2, 3])`. Be sure to also increase `max_evals` in the `fmin` call below.
# MAGIC
# MAGIC See https://docs.databricks.com/applications/machine-learning/automl-hyperparam-tuning/index.html
# MAGIC for more information on hyperparameter tuning, as well as
# MAGIC http://hyperopt.github.io/hyperopt/getting-started/search_spaces/ for documentation on supported
# MAGIC search expressions.
# MAGIC
# MAGIC For documentation on parameters used by the model in use, please see:
# MAGIC https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html
# MAGIC
# MAGIC NOTE: The above URL points to a stable version of the documentation corresponding to the last
# MAGIC released version of the package. The documentation may differ slightly for the package version
# MAGIC used by this notebook.

# COMMAND ----------

# Run with maybe 2 evals
# Run with maybe 5 evals if wanting more examples
space = {
    "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1.0),
    "lambda_l1": hp.loguniform("lambda_l1", -2, 3),  # ~0.1 to ~20
    "lambda_l2": hp.loguniform("lambda_l2", -2, 5),
    "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
    "max_bin": hp.quniform("max_bin", 128, 512, 1),
    "max_depth": hp.quniform("max_depth", 4, 12, 1),
    "min_child_samples": hp.quniform("min_child_samples", 20, 100, 1),
    "n_estimators": hp.quniform("n_estimators", 100, 400, 10),
    "num_leaves": hp.quniform("num_leaves", 20, 150, 5),
    "path_smooth": hp.uniform("path_smooth", 1, 100),
    "subsample": hp.uniform("subsample", 0.5, 1.0),
    "random_state": 42,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run trials
# MAGIC When widening the search space and training multiple models, switch to `SparkTrials` to parallelize
# MAGIC training on Spark:
# MAGIC ```
# MAGIC from hyperopt import SparkTrials
# MAGIC trials = SparkTrials()
# MAGIC ```
# MAGIC
# MAGIC NOTE: While `Trials` starts an MLFlow run for each set of hyperparameters, `SparkTrials` only starts
# MAGIC one top-level run; it will start a subrun for each set of hyperparameters.
# MAGIC
# MAGIC See http://hyperopt.github.io/hyperopt/scaleout/spark/ for more info.

# COMMAND ----------

trials = Trials()
fmin(objective,
     space=space,
     algo=tpe.suggest,
     max_evals= 2, # Increase this when widening the hyperparameter search space.
     trials=trials)

best_result = trials.best_trial["result"]
model = best_result["model"]
mlflow_run = best_result["run"]

display(
  pd.DataFrame(
    [best_result["val_metrics"], best_result["test_metrics"]],
    index=["validation", "test"]))

# COMMAND ----------

# MAGIC %md
# MAGIC # Register the best model

# COMMAND ----------

from mlflow import register_model
from mlflow.tracking import MlflowClient
import mlflow

# Get active run ID
run_id = mlflow_run.info.run_id
model_uri = f"runs:/{run_id}/model"

# Register the model
registration = register_model(
    model_uri=model_uri,
    name=model_name  # Must be Unity Catalog path like dev.schema.model_name
)

# Wait until model is ready
client = MlflowClient()
import time
while client.get_model_version(name=model_name, version=registration.version).status != "READY":
    time.sleep(1)

# Assign alias
alias = "champion" if registration.version == '1' else "challenger"

client.set_registered_model_alias(
    name=model_name,
    alias=alias,
    version=registration.version
)

print(f"Registered model version {registration.version} with alias '{alias}'")


# COMMAND ----------

# DBTITLE 1,Apply tags to help document
from mlflow.tracking import MlflowClient

client = MlflowClient()

client.set_model_version_tag(
    name=model_name,
    version=registration.version,
    key="model_type",
    value=type(model).__name__  # e.g., LightGBM Model
)

client.set_model_version_tag(
    name=model_name,
    version=registration.version,
    key="modeling_method",
    value="Custom Model"
)

# We can also tag the model version with the F1 score for visibility
client.set_model_version_tag(
  name=model_name,
  version=registration.version,
  key="best_run_id",
  value=run_id
)

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
