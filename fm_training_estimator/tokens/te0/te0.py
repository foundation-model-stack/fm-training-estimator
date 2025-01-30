# Third Party
from datasets import load_dataset
from tqdm import tqdm
import numpy as np

# Local
from ...config import DataArguments
from ...utils import logger
from ..te import TokenEstimator

RUNS = 5
SEED = 42

np.random.seed(SEED)


class TokenEstimator0(TokenEstimator):
    def __init__(self, da: DataArguments):
        if da.dataset is None:
            raise RuntimeError("Dataset argument has to be filled in for TE0!")

        if da.dataset.endswith(".json") or da.dataset.endswith(".jsonl"):
            logger.debug("Tokens TE0 - Parsing dataset as local json file")
            dataset = load_dataset("json", data_files={"train": da.dataset})["train"]
        else:
            dataset = load_dataset(
                da.dataset,
                name=da.dataset_config_name,
                split=da.dataset_split,
                trust_remote_code=da.trust_remote_code
            )

        tokens = []
        logger.info("Tokens TE0 - Loading data in dataset...")
        for item in tqdm(dataset):
            txt = da.dataset_text_field.format_map(item)
            tokens.append(int(len(txt) / 3.6))

        self.tokens = tokens

    def get_total_tokens(self):
        return np.sum(self.tokens)

    def get_estimated_batch_width(self, batch_size, runs=RUNS):
        widths = [
            self.get_estimated_batch_width_random_shuffle(batch_size)
            for i in range(runs)
        ]
        return np.mean(widths)

    def get_num_samples(self):
        return len(self.tokens)

    def get_estimated_batch_width_random_shuffle(self, bs):
        tokens = np.array(self.tokens)
        np.random.shuffle(tokens)
        if len(tokens) % bs != 0:
            tokens = np.concatenate(
                [tokens, np.zeros(bs - len(tokens) % bs)]
            )  # simulating drop_last=False
        return np.mean(np.max(np.split(tokens, len(tokens) / bs), axis=1))

    def get_max_sample_length(self):
        return np.max(self.tokens)
