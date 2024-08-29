# Third Party
from xgboost import XGBRegressor
import pandas

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

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path):
        self.model.load_model(model_path)

    def train(self, data_path: str, model_path: str, y_headers: list[str]):
        data = pandas.read_csv(data_path)

        # obtain the data format metadata
        data_keys = ",".join(list(data.columns.values))

        for col in data:
            if data[col].dtype == object:
                data[col] = data[col].astype("category")

        X = data.drop(columns=y_headers)
        Y = data[y_headers]

        self.model.fit(X, Y)

        # save the feature names into the model
        self.model.get_booster().feature_names = list(X.columns)
        # save the data format
        self.model.get_booster().set_attr(
            data_format_version=lookup_format_version(data_keys)
        )

        # save model to file
        self.model.save_model(model_path)

    def run(self, X):
        # convert input data array into form suitable to feed in

        # add column names
        data = pandas.DataFrame([X], columns=self.model.get_booster().feature_names)

        # setup category column types
        for col in data:
            if data[col].dtype == object:
                data[col] = data[col].astype("category")

        return self.model.predict(data)

    def get_data_format(self):
        return self.model.get_booster().attr("data_format_version")
