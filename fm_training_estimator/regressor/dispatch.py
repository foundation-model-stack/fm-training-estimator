import zipfile

from .xgboost import XGBoostRegressor
from .linear import LinearRegressor
from .arise import AriseRegressor

def GetRegressor(model_path):
    with zipfile.ZipFile(model_path, mode='r') as model_zip:
        mt = model_zip.read("model_type").decode()

        if mt == "linear":
            return LinearRegressor(model_path)
        elif mt == "xgboost":
            return XGBoostRegressor(model_path)
        elif mt == "arise":
            return AriseRegressor(model_path)
        else:
            raise ValueError("Unknown model type found", mt)
