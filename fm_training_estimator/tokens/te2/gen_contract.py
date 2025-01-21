import json

# Third Party
import fire

# Local
from .te2 import GenerateTokenEstimator2Contract

def gen(dataset: str, output: str, ds_config_name: str = None, ds_split: str = "test"):

    print("Generating contract...")
    contract = GenerateTokenEstimator2Contract(dataset, ds_config_name, ds_split)

    with open(output, "w") as f:
        json.dump(contract, f)

    print("...successfully wrote contract to file: ", output)

if __name__ == "__main__":
    fire.Fire(gen)
