import os
import zipfile
import tempfile

# Third Party
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

# Local
from ...data import lookup_format_version


class LinearRegressor:
    def __init__(self, model_path=None):
        self.model = RandomForestRegressor()
        self.cat_enc = OneHotEncoder(sparse_output=False).set_output(transform="pandas")

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path):
        with tempfile.TemporaryDirectory() as mdir:
            with zipfile.ZipFile(model_path) as model_zip:
                model_zip.extractall(mdir)

            path_m = os.path.join(mdir, "model.json")
            self.model = joblib.load(path_m)

            path_e = os.path.join(mdir, "cat_enc.json")
            self.cat_enc = joblib.load(path_e)

    def train(self, data_path: str, model_path: str, y_headers: list[str]):
        data = pandas.read_csv(data_path)

        # prepare the pure list of X for saving as metadata later
        X_cols = list(data.drop(columns=y_headers).columns)

        # obtain the data format metadata
        data_keys = ",".join(list(data.columns.values))

        # encode category columns
        cat_feats = data.dtypes[data.dtypes=='object'].index.values.tolist()
        ecats = self.cat_enc.fit_transform(data[cat_feats])

        data = data.drop(columns=cat_feats)
        data = pandas.concat([data, ecats], axis=1)

        X = data.drop(columns=y_headers)
        Y = data[y_headers]

        self.model.fit(X, Y)

        # save the feature names into the model
        self.model.metadata = {}
        self.model.metadata['feature_names'] = X_cols
        # save the data format
        self.model.metadata['data_format_version']=lookup_format_version(data_keys)

        with ( tempfile.NamedTemporaryFile(suffix='.json', mode='w') as buf_m,
               tempfile.NamedTemporaryFile(suffix='.json', mode='w') as buf_e,
               tempfile.NamedTemporaryFile(mode='w') as buf_mt,
               zipfile.ZipFile(model_path, mode='w') as model_zip
              ):

            # save model to tmp buffer
            joblib.dump(self.model, buf_m.name)
            # save encoder into tmp buffer
            joblib.dump(self.cat_enc, buf_e.name)
            # save model type to file
            with open(buf_mt.name, 'w') as f:
                f.write("linear")

            # now move the files into the zip file
            model_zip.write(buf_m.name, 'model.json')
            model_zip.write(buf_e.name, 'cat_enc.json')
            model_zip.write(buf_mt.name, 'model_type')

    def run(self, X):
        # convert input data array into form suitable to feed in

        # add column names
        data = pandas.DataFrame([X], columns=self.model.metadata['feature_names'])

        # encode category columns
        cat_feats = data.dtypes[data.dtypes=='object'].index.values.tolist()
        ecats = self.cat_enc.transform(data[cat_feats])

        data = data.drop(columns=cat_feats)
        data = pandas.concat([data, ecats], axis=1)

        return self.model.predict(data)

    def get_data_format(self):
        return self.model.metadata["data_format_version"]
