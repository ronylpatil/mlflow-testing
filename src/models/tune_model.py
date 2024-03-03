import mlflow
import yaml
import pathlib
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
from src.data.make_dataset import load_data
from functools import partial
from src.visualization import visualize

def objective(params: dict, yaml_obj: dict, x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, plots_dir: str) -> dict :
     mlflow_config = yaml_obj['mlflow_config']
     remote_server_uri = mlflow_config['remote_server_uri']
     exp_name = mlflow_config['tunningExpName']
     mlflow.set_tracking_uri(remote_server_uri)
     mlflow.set_experiment(experiment_name = exp_name)
     # adding experiment description
     experiment_description = ('tracking model tunning process. Using HYPEROPT to tune the machine learning model.') 
     mlflow.set_experiment_tag("mlflow.note.content", experiment_description)

     with mlflow.start_run(description = 'tunning random forest model using hyperopt optimization technique') :
          mlflow.set_tags({'project_name': 'wine-quality', 'author' : 'ronil', 'project_quarter': 'Q1-2024'})
          mlflow.log_params(params)
          
          model = RandomForestClassifier(**params)
          model.fit(x_train, y_train)

          y_pred = model.predict(x_test)
          y_pred_prob = model.predict_proba(x_test)
          accuracy = metrics.balanced_accuracy_score(y_test, y_pred)
          precision = metrics.precision_score(y_test, y_pred, zero_division = 1, average = 'macro')
          recall = metrics.recall_score(y_test, y_pred, average = 'macro')
          roc_score = metrics.roc_auc_score(y_test, y_pred_prob, average = 'macro', multi_class = 'ovr')

          filename = visualize.conf_matrix(y_test, y_pred, model.classes_, path = plots_dir, params_obj = yaml_obj)
          mlflow.log_artifact(filename, 'confusion_matrix')
          mlflow.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall, "roc_score": roc_score})

     return {'loss': -accuracy, 'status': STATUS_OK}

search_space = { 'n_estimators': hp.randint('n_estimators', 200 - 15) + 15,
                 'criterion': hp.choice('criterion', ['gini', 'entropy']),
                 'max_depth': hp.randint('max_depth', 100 - 5) + 5,
                 'min_samples_split': hp.randint('min_samples_split', 100 - 5) + 5,
                 'min_samples_leaf': hp.randint('min_samples_leaf', 100 - 5) + 5   
               }

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent

params = yaml.safe_load(open(f'{home_dir.as_posix()}/params.yaml'))
cm_dir = f'{home_dir.as_posix()}/plots/tunning_plots'

parameters = params['train_model']
TARGET = params['base']['target']

train_data = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_train.csv"
test_data = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_test.csv"

train_data = load_data(train_data)
x_train = train_data.drop(columns = [TARGET]).values
y_train = train_data[TARGET]

test_data = load_data(test_data)
x_test = test_data.drop(columns = [TARGET]).values
y_test = test_data[TARGET]

additional_params = {'yaml_obj': params, 'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test,
                     'plots_dir': cm_dir}

partial_obj = partial(objective, **additional_params)

best_result = fmin(fn = partial_obj,
                    space = search_space,
                    algo = tpe.suggest,
                    max_evals = 1,
                    trials = Trials()
              )
