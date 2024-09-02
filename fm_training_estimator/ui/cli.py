# Standard
from pathlib import Path
from typing import Optional
import json
import logging

# Third Party
import fire

# Local
from .core import run


def run_cli(
    config: str,
    output_path: str = "",
    log_level: str = "INFO",
    lookup_data_path: Optional[str] = None,
    model_path: Optional[str] = None,
    use_model_features: bool = True,
):
    """Run the CLI."""
    log_level = log_level.upper()
    logging.basicConfig(level=log_level)
    output = run(
        config=config,
        lookup_data_path=lookup_data_path,
        model_path=model_path,
        use_model_features=use_model_features,
    )
    output_dict = {
        "total_mem_estimate_og": float(output["total_mem_estimate_og"]),
        "activation_memory_og": float(output["activation_memory_og"]),
        "gradient_memory_og": float(output["gradient_memory_og"]),
        "model_memory_og": float(output["model_memory_og"]),
        "optimizer_memory_og": float(output["optimizer_memory_og"]),
        "total_mem_estimate": output["total_mem_estimate"],
        "activation_memory": output["activation_memory"],
        "gradient_memory": output["gradient_memory"],
        "model_memory": output["model_memory"],
        "optimizer_memory": output["optimizer_memory"],
        "tps": float(output["tps"]),
    }
    output_json = json.dumps(output_dict, indent=4)
    if output_path == "":
        # use print instead of logging so that
        # the output can be parsed as valid json
        print(output_json)
        return
    output_path: Path = Path(output_path)
    output_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    logging.info("writing the output to a file at %s", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_json)


if __name__ == "__main__":
    fire.Fire(run_cli)
