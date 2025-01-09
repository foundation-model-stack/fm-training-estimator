import os
import tempfile
import shutil
import pandas
import yaml

from arise_predictions.sdk import execute_auto_build_models, get_base_args

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
