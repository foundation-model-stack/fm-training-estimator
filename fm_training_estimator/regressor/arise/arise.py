import os
import tempfile
import shutil
import pandas
import yaml
import zipfile
from functools import lru_cache

from arise_predictions.preprocessing import job_parser
from arise_predictions.utils import constants, utils
from arise_predictions.auto_model.build_models import auto_build_models, get_estimators_config
from arise_predictions.perform_predict.predict import demo_predict, get_predict_config_from_dict

from ...data import lookup_format_version, get_format_by_version

class AriseRegressor:
    def __init__(self, model_path=None):
        self.model_path = model_path

    def preprocess(self, workdir, job_spec):
        inputs = sorted(list(job_spec[0]))
        outputs = sorted(list(job_spec[1]))
        start_time_field_name = job_spec[2]
        end_time_field_name = job_spec[3]
        job_parser_class_name = job_spec[4]
        job_entry_filter = job_spec[5]
        feature_engineering = job_spec[6] if len(job_spec) > 6 else None
        metadata_parser_class_name = job_spec[7] if len(job_spec) > 7 else None
        history_file = os.path.join(workdir, constants.JOB_HISTORY_FILE_NAME + ".csv")

        history_data, history_file = job_parser.collect_jobs_history(
                os.path.join(workdir, constants.JOB_DATA_DIR), workdir, inputs, outputs,
                start_time_field_name, end_time_field_name, None, job_parser_class_name,
                job_entry_filter, feature_engineering, metadata_parser_class_name,
                workdir)
        return history_data, history_file

    def execute_build(self, workdir, js, config_path):

        history_data, history_file = self.preprocess(workdir, js)
        outputs = sorted(list(js[1]))
        output_path = os.path.join(workdir, constants.AM_OUTPUT_PATH_SUFFIX)

        auto_build_models(raw_data=history_data,
                          config=get_estimators_config(config_path, num_jobs=-1),
                          target_variables=outputs,
                          output_path=output_path,
                          leave_one_out_cv=None,
                          feature_col=None,
                          low_threshold=None,
                          high_threshold=None,
                          single_output_file=True,
                          randomized_hpo=False,
                          n_random_iter=False)

    def train(self, data_path: str, model_path: str, config_path: str, y_headers: list[str]):
        with tempfile.TemporaryDirectory() as workdir:
            print(workdir)

            datadir = os.path.join(workdir, "data")
            os.mkdir(datadir)
            shutil.copy2(data_path, datadir)

            # we only need the headers, so we read just a single row
            data = pandas.read_csv(data_path, nrows=1)
            # these 2 lines are for calc data version needed by manager module
            data_keys = ",".join(list(data.columns.values))
            data_version = lookup_format_version(data_keys)
            # this is used for arise
            x_headers = list(set(data.columns.values) - set(y_headers))

            # prepare the job spec file
            job_spec = {"job-metadata-inputs": {}, "job-metadata-outputs": y_headers}
            for h in x_headers:
                job_spec["job-metadata-inputs"][h] = 0

            js = job_parser.parse_job_spec(job_spec)

            # pre-emptively create output dir for arise
            output_path = os.path.join(workdir, constants.AM_OUTPUT_PATH_SUFFIX)
            utils.mkdirs(output_path)
            # normally, arise saves job spec into the model file
            # but, we skip it here
            # save the data version into a file in arise model
            with open(os.path.join(output_path, "estimator_data_version"), "w") as f:
                f.write(data_version)

            # save the model type also here
            with open(os.path.join(output_path, "model_type"), "w") as f:
                f.write("arise")

            self.execute_build(workdir, js, config_path)

            # copy the model to required destination
            shutil.copy2(os.path.join(workdir, "ARISE-auto-models.zip"), model_path)

    def get_columns(self):
        col_str = get_format_by_version(self.get_data_format()).X
        return col_str.split(",")

    def execute_predict(self, workdir, js, predict_config, model_file_name):
        return demo_predict(
            original_data=None,
            config=get_predict_config_from_dict(predict_config),
            estimator_path=model_file_name,
            feature_engineering=js[6],
            metadata_parser_class_name=js[7],
            metadata_path=model_file_name,
            output_path=os.path.join(workdir, constants.PRED_OUTPUT_PATH_SUFFIX))

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

            job_spec = {"job-metadata-inputs": {}, "job-metadata-outputs": [y]}
            for h in cols:
                job_spec["job-metadata-inputs"][h] = 0

            shutil.copy2(self.model_path, workdir)
            model_file_name = os.path.join(workdir, os.path.basename(self.model_path))
 
            js = job_parser.parse_job_spec(job_spec)
            self.execute_predict(workdir, js, predict_config, model_file_name)

            # now read the result
            res_file = os.path.join(workdir, "ARISE-predictions", "all-predictions.csv")
            res = pandas.read_csv(res_file)

            return res[y][0]

    @lru_cache
    def get_data_format(self):
        with zipfile.ZipFile(self.model_path) as model_zip:
            with model_zip.open("estimator_data_version", 'r') as edv:
                dv = edv.read().decode()

        return dv
