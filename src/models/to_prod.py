# search the top-5 models having minimum error or maximum accuracy
# tag them alias
# if any alias name "produciton" already available, put it into archieve and assign prod alias to other model
 
import pathlib
import yaml
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint

'''
# curr_dir = pathlib.Path(__file__)
# home_dir = curr_dir.parent.parent.parent

# params = yaml.safe_load(open(f'{home_dir.as_posix()}/params.yaml'))
# MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# client = MlflowClient(tracking_uri = MLFLOW_TRACKING_URI)

# # get list of all experiments (active only, deleted, all)

# # print(client.search_experiments())
# # for i in client.search_experiments() : 
#      # print(i.name, '   ', i.experiment_id)

# exp_name = 'tunningExperiments'

# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# # for i in mlflow.search_experiments(): print(i.name)

# for i in client.search_experiments(): print(i.name)
# # ye experiment name se uski exp id dega
# exp_id = [int(i.experiment_id) for i in client.search_experiments(1) if i.name == exp_name][0]


# # ab is experiments id se runs search kr jo kafi jyada efficient ho
# from mlflow.entities import ViewType
# runs = client.search_runs(experiment_ids = exp_id, filter_string = 'metrics.accuracy > 0.75', 
#                           run_view_type = ViewType.ACTIVE_ONLY, order_by = ['metrics.accuracy DESC'])

# # yha se bhi hum model ko production me dal skte he
# # ya UI se bhi, UI wala option simple he

# for i in runs :
#      print(f'run_id: {i.info.run_id}   run_name: {i.info.run_name}   accuracy: {i.data.metrics["accuracy"]}')


# run_id = runs[0].info.run_id
# print(run_id)

# mlflow.register_model(model_uri = 'runs:/1045a86af727432d8c830c559091c74a/model/', name = 'test', tags = {'status': 'intermediate'})
'''

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent

params = yaml.safe_load(open(f'{home_dir.as_posix()}/params.yaml'))
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"

# client = MlflowClient(tracking_uri = MLFLOW_TRACKING_URI)

# ----------------- tracking URI
# In this approach, you explicitly set the tracking URI using mlflow.set_tracking_uri() before initializing the
# MlflowClient object. Then, when you create the MlflowClient, you explicitly pass the tracking_uri parameter with 
# the same URI you set earlier. This approach ensures that the client uses the specified tracking URI for all
# subsequent operations.

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri = MLFLOW_TRACKING_URI)
# client = MlflowClient()

# for i in mlflow.search_experiments(): print(i.name)

# seach register model by model name 
reg_model_name = 'prod_testing'
tag = 'validation_status'
tag_value = 'staging'

# print(f'searching reg model having name {reg_model_name} and tag : [{tag} : {tag_value}]')
# ye aliases nahi bata rah he(bug he), but ye result de rha he pura
# print(client.search_model_versions(filter_string = "name = 'prod_testing' and tag.validation_status = 'pending'"))

# filter the runs of registerded model based on tag
# for i in client.search_registered_models() : 
     # pprint(i)

# ye exp k run search kr ra he
# pprint(client.search_runs(experiment_ids = 3)[0])


# working condition
prod = client.get_model_version_by_alias(name = reg_model_name, alias = 'staging')
print("Model Name: ", prod.name)
print("Model Version: ", prod.version)
print("Current State: ", prod.aliases[0])
print("Run ID: ", prod.run_id)


from mlflow.sklearn import load_model

model = load_model(f"models:/{prod.name}/{prod.version}")

inp = [[8.207097522651933,0.3651906838774485,0.378633478714214,4.542055303465597,
               0.079792372644902,25.448093161225515,38.8633478714214,0.9945203178994676,
               3.23103813677549,0.6893432309794074,12.555190683877449,8.950921685243596,
               2.7702927995076014,0.6548095970892682,1.4026701523459664,11.995915730668555,
               12.624368208379435,15.786228820652939,5.311190907089906]]

print(model.predict(inp))
print(model.classes_)
print(max(model.predict_proba(inp)[0]))


# ye registered model ki thodi details dega
# pprint(client.search_registered_models())



# here we are not getting alias this is bug in this method in mlflow
# print(client.search_model_versions(filter_string = "name = 'prod_testing'"))



