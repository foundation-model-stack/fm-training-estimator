# Third Party
import fire

# Local
from .xgboost import XGBoostRegressor


def train(data_path: str, model_path: str, y_headers: list[str]):
    model = XGBoostRegressor()

    if not model_path.endswith(".json"):
        print("model_path must be a json extension.")
        print("Refusing to continue!!")
        return

    print("Training model...")
    model.train(data_path, model_path, y_headers)
    print("...successfully wrote model to file: ", model_path)


if __name__ == "__main__":
    fire.Fire(train)
