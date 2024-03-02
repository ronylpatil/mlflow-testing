import pathlib
import joblib
import yaml
import mlflow
import pandas as pd
import numpy as np
from typing import IO, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn import metrics
from src.logger import infologger
from dvclive import Live

infologger.info('*** Executing: predict_model.py ***')
from src.data.make_dataset import load_data

def load_model(model_dir: str) -> BaseEstimator :
     try : 
          model = joblib.load(model_dir)
     except Exception as e : 
          infologger.info(f'there\'s an issue while loading the model from {model_dir} [check load_model()]. exc: {e}')
     else : 
          infologger.info(f'model loaded successfully from {model_dir}')
          return model


def evaluate(x_test: pd.DataFrame, y_test: np.ndarray, model: BaseEstimator, yaml_file_obj: IO[Any]) -> None : 
     try : 
          y_pred = model.predict(x_test)     # return class
          y_pred_prob = model.predict_proba(x_test)    # return probability
     except Exception as oe : 
          infologger.info(f'there\'s an issue while prediction [check evaluate()]. exc: {oe}')
     else : 
          try : 
               bal_acc = metrics.balanced_accuracy_score(y_test, y_pred)
               precision = metrics.precision_score(y_test, y_pred, zero_division = 1, average = 'macro')
               recall = metrics.recall_score(y_test, y_pred, average = 'macro')
               roc_score = metrics.roc_auc_score(y_test, y_pred_prob, average = 'macro', multi_class = 'ovr')
          except Exception as e :
               infologger.info(f'there\'s an issue while evalution [check evaluate()]. exc: {e}')
          else : 
               infologger.info('model evalution done')
               try : 
                    mlflow_config = yaml_file_obj['mlflow_config']
                    remote_server_uri = mlflow_config['remote_server_uri']
                    mlflow.set_tracking_uri(remote_server_uri)
                    mlflow.set_experiment(mlflow_config['testingExpName'])

                    with mlflow.start_run(run_name = mlflow_config['testingRunName']) : 
                         mlflow.set_tag('tag', 'v1')

                         mlflow.log_metric('bal_accuracy', float('{:.2f}'.format(bal_acc)))
                         mlflow.log_metric('roc_score', float('{:.2f}'.format(roc_score)))
                         mlflow.log_metric('precision', float("{:.2f}".format(precision)))
                         mlflow.log_metric('recall', float("{:.2f}".format(recall)))
                         

               except Exception as ie : 
                    infologger.info(f'there\'s an issue while tracking the testing performance metrics [check evaluate()]. exc: {ie}')
               else :
                    infologger.info('performance metrics tracked by mlflow')

def main() -> None : 
     curr_dir = pathlib.Path(__file__)
     home_dir = curr_dir.parent.parent.parent
     try : 
          params = yaml.safe_load(open(f'{home_dir.as_posix()}/params.yaml', encoding = 'utf8'))
     except Exception as e :
          infologger.info(f'there\'s an issue while loading params.yaml [check main()]. exc: {e}')
     else :
          data_dir = f"{home_dir.as_posix()}{params['build_features']['extended_data']}/extended_test.csv"
          model_dir = f'{home_dir.as_posix()}{params["train_model"]["model_dir"]}/model.joblib'
          test_data = load_data(data_dir)
          TARGET = params['base']['target']
          x_test = test_data.drop(columns = [TARGET]).values
          y_test = test_data[TARGET]

          evaluate(x_test, y_test, load_model(model_dir), yaml_file_obj = params)
          infologger.info('program terminated normally')

if __name__ == '__main__' :
     infologger.info('predict_model.py as __main__')
     main()
