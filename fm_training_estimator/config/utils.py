# First Party
from fm_training_estimator.config.arguments import HFTrainingArguments


def is_fsdp(ta: HFTrainingArguments):
    if hasattr(ta, "fsdp") and len(ta.fsdp) != 0:
        return True

    return False
