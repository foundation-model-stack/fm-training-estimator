# Third Party
import fire

# Local
from .arise import AriseRegressor

def train(data_path: str, model_path: str, config_path: str, y_headers: list[str]):
    """Train a AriseRegressor model that can be used by this estimator library.

    Args:
        data_path (str): the path to training data
        model_path (str): the output path of trained model. Must end with .json.
        config_path (str): the path to an ARISE training config file. See: https://github.com/arise-insights/arise-predictions/tree/main/config for examples.
        y_headers (list[str]): list of column names to drop from data

    """
    model = AriseRegressor()

    if not model_path.endswith(".zip"):
        print("model_path must be a zip extension.")
        print("Refusing to continue!!")
        return

    print("Training model...")
    model.train(data_path, model_path, config_path, y_headers)
    print("...successfully wrote model to file: ", model_path)


if __name__ == "__main__":
    fire.Fire(train)
