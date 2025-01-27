import json

# Third Party
import fire

# Local
from .te2 import GenerateTokenEstimator2Contract

def gen(dataset: str, output: str, ds_config_name: str = None, ds_split: str = "test", sample_percent: int = None):
    """
    Inputs: 
    dataset: <str> the path to a json/jsonl file, or the name of HF dataset on the HF hub
    output: <str> the path to output the contract file
    ds_config_name: <str> For HF datasets, optional name of config to use
    ds_split: <str> for HF datasets, the name of the split of the data to use
    sample_percent: <int> an optional integer between (0-100], indicating what percent of the dataset we should sample. Default (if nothing specified), is no sampling, which means 100% of the data is used.
    """

    print("Generating contract...")
    contract = GenerateTokenEstimator2Contract(dataset, ds_config_name, ds_split, sample_percent)

    with open(output, "w") as f:
        json.dump(contract, f)

    print("...successfully wrote contract to file: ", output)

if __name__ == "__main__":
    fire.Fire(gen)
