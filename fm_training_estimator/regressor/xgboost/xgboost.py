import os
import zipfile
import tempfile

# Third Party
from xgboost import XGBRegressor
import pandas
from sklearn.preprocessing import OrdinalEncoder
import joblib

# Local
from ...data import lookup_format_version


class XGBoostRegressor:
    def __init__(self, model_path=None):
        self.model = XGBRegressor(
            n_estimators=400,
            max_depth=7,
            eta=0.1,
            subsample=0.7,
            colsample_bytree=0.8,
            enable_categorical=True,
        )
        self.cat_enc = OrdinalEncoder()

        if model_path is not None:
            self.load(model_path)


    def load(self, model_path):
        with tempfile.TemporaryDirectory() as mdir:
            with zipfile.ZipFile(model_path) as model_zip:
                model_zip.extractall(mdir)

            path_m = os.path.join(mdir, "model.json")
            self.model.load_model(path_m)

            path_e = os.path.join(mdir, "cat_enc.json")
            self.cat_enc = joblib.load(path_e)

    def train(self, data_path: str, model_path: str, y_headers: list[str]):
        data = pandas.read_csv(data_path)

        # obtain the data format metadata
        data_keys = ",".join(list(data.columns.values))

        # ordinal encode all "object" type columns, which are actually categories
        cat_feats = data.dtypes[data.dtypes=='object'].index.values.tolist()
        data[cat_feats] = self.cat_enc.fit_transform(data[cat_feats])

        # now mark these as categorical feats
        for cf in cat_feats:
            data[cat_feats] = data[cat_feats].astype("category")

        X = data.drop(columns=y_headers)
        Y = data[y_headers]

        self.model.fit(X, Y)

        # save the feature names into the model
        self.model.get_booster().feature_names = list(X.columns)
        # save the data format
        self.model.get_booster().set_attr(
            data_format_version=lookup_format_version(data_keys)
        )

        with ( tempfile.NamedTemporaryFile(suffix='.json', mode='w') as buf_m,
               tempfile.NamedTemporaryFile(suffix='.json', mode='w') as buf_e,
               tempfile.NamedTemporaryFile(mode='w') as buf_mt,
               zipfile.ZipFile(model_path, mode='w') as model_zip
              ):

            # save model to tmp buffer
            self.model.save_model(buf_m.name)
            # save encoder into tmp buffer
            joblib.dump(self.cat_enc, buf_e.name)
            # save model type to file
            with open(buf_mt.name, 'w') as f:
                f.write("xgboost")

            # now move the files into the zip file
            model_zip.write(buf_m.name, 'model.json')
            model_zip.write(buf_e.name, 'cat_enc.json')
            model_zip.write(buf_mt.name, 'model_type')


    def run(self, X):
        # convert input data array into form suitable to feed in

        # add column names
        data = pandas.DataFrame([X], columns=self.model.get_booster().feature_names)

        # encode category columns
        cat_feats = data.dtypes[data.dtypes=='object'].index.values.tolist()
        data[cat_feats] = self.cat_enc.transform(data[cat_feats])

        # now mark these as categorical feats
        for cf in cat_feats:
            data[cat_feats] = data[cat_feats].astype("category")

        return self.model.predict(data)

    def get_data_format(self):
        return self.model.get_booster().attr("data_format_version")
