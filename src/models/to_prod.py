# search the top-5 models having minimum error or maximum accuracy
# tag them alias
# if any alias "produciton" already available, put it into archieve and assign prod alias to other model
 
# import pathlib
# import yaml
# import mlflow
# from mlflow.tracking import MlflowClient
# from pprint import pprint

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
# for i in mlflow.search_experiments(): print(i.name)
# ye experiment name se uski exp id dega
# exp_id = [int(i.experiment_id) for i in client.search_experiments(1) if i.name == exp_name][0]


# ab is experiments id se runs search kr jo kafi jyada efficient ho
# from mlflow.entities import ViewType
# runs = client.search_runs(experiment_ids = exp_id, filter_string = 'metrics.accuracy > 0.75', 
                         #  run_view_type = ViewType.ACTIVE_ONLY, order_by = ['metrics.accuracy DESC'])

# yha se bhi hum model ko production me dal skte he
# ya UI se bhi, UI wala option simple he

# for i in runs :
     # print(f'run_id: {i.info.run_id}   run_name: {i.info.run_name}   accuracy: {i.data.metrics["accuracy"]}')


# run_id = runs[0].info.run_id
# print(run_id)

# mlflow.register_model(model_uri = 'runs:/20bf8d42ba3449039c77261a9a835a15/model/', name = 'luxuriant-goat-351', tags = {'status': 'intermediate'})


# client.set_registered_model_alias('wineq', 'production', )




acc = 6.23568475

print(f'hello: {acc:.3f}')