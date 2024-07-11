# Local
from .arguments import TrainingArguments


def is_fsdp(ta: TrainingArguments):
    if hasattr(ta, "fsdp") and len(ta.fsdp) != 0:
        return True

    return False
