from .xgboost import XGBoostRegressor
from .linear import LinearRegressor

def GetRegressor(model_path):
    return LinearRegressor(model_path)
