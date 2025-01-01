# Standard
import logging

# Local
from ..config import HFTrainingArguments, InfraArguments
from ..tokens import TokenEstimator

MODEL_LOAD_TIME = 5 * 60
CHECKPOINT_TIME = 60


def get_total_time(
    hf: HFTrainingArguments, ia: InfraArguments, te: TokenEstimator, tps, tokens
):
    train_time = tokens / tps

    num_epochs = hf.num_train_epochs

    # one checkpoint at the very end
    num_checkpoints = 1
    ss = hf.save_strategy
    if ss == "epoch":
        num_checkpoints += num_epochs
    elif ss == "steps":
        steps_in_epoch = te.get_num_samples() / (
            hf.per_device_train_batch_size * ia.numGpusPerPod
        )
        num_checkpoints += num_epochs * steps_in_epoch / hf.save_steps
    elif ss == "best":
        logging.warn(
            "Unable to guess number of checkpoints due to use of `best` saving strategy."
        )

    return MODEL_LOAD_TIME + train_time * num_epochs + num_checkpoints * CHECKPOINT_TIME
