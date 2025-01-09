import os
import tempfile
import shutil
import pandas
import yaml

from arise_predictions.sdk import execute_auto_build_models, get_base_args, execute_predict
from ...data import get_format_by_version

class AriseRegressor:
    def __init__(self, model_path=None):
        self.model_path = model_path

    def train(self, data_path: str, model_path: str, config_path: str, y_headers: list[str]):
        with tempfile.TemporaryDirectory() as workdir:
            print(workdir)

            datadir = os.path.join(workdir, "data")
            os.mkdir(datadir)
            shutil.copy2(data_path, datadir)

            # we only need the headers, so we read just a single row
            data = pandas.read_csv(data_path, nrows=1)
            x_headers = list(set(data.columns.values) - set(y_headers))

            # prepare the job spec file
            job_spec = {"job-metadata-inputs": {}, "job-metadata-outputs": y_headers}
            for h in x_headers:
                job_spec["job-metadata-inputs"][h] = 0
            job_file = os.path.join(workdir, "job_spec.yaml")
            with open(job_file, "w") as jobfile:
                yaml.dump(job_spec, jobfile)

            args = get_base_args("auto-build-models")
            args.input_path = workdir
            args.reread_history = True
            args.single_output_file = True
            args.config_file = config_path

            execute_auto_build_models(args)

            # copy the model to required destination
            shutil.copy2(os.path.join(workdir, "ARISE-auto-models.zip"), model_path)

    def get_columns(self):
        # TODO: return data from the job spec file loaded from the model zip itself 
        col_str = get_format_by_version(self.get_data_format()).X
        return col_str.split(",")

    def run(self, X, y):
        cols = self.get_columns()
        input_vars = []
        for k, v in zip(cols, X):
            input_vars.append({k: v})

        with tempfile.TemporaryDirectory() as workdir:
            # TODO: get the right value for the "greater is better" variable
            est_config = {"target_variable": y, "greater_is_better": True}
            predict_config = {"fixed_values": input_vars,
                              "variable_values": [],
                              "estimators": [est_config]}

            predict_file = os.path.join(workdir, "predict-config.yaml")
            with open(predict_file, "w") as pfile:
                yaml.dump(predict_config, pfile)

            job_spec = {"job-metadata-inputs": {}, "job-metadata-outputs": [y]}
            for h in cols:
                job_spec["job-metadata-inputs"][h] = 0
            job_file = os.path.join(workdir, "job_spec.yaml")
            with open(job_file, "w") as jobfile:
                yaml.dump(job_spec, jobfile)

            shutil.copy2(self.model_path, workdir)
            model_file_name = os.path.join(workdir, os.path.basename(self.model_path))
 
            args = get_base_args("predict")
            args.input_path = workdir
            args.config_file = predict_file
            args.model_path = model_file_name

            execute_predict(args)

            # now read the result
            res_file = os.path.join(workdir, "ARISE-predictions", "all-predictions.csv")
            res = pandas.read_csv(res_file)

            return res[y][0]

    def get_data_format(self):
        return "v3"
        # TODO: get it from the model file
        # return self.model.get_booster().attr("data_format_version")
