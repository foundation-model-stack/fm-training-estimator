# Copyright The FM Training Estimator Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script wraps fm_training_estimator to run with user provided training configs.
The script will read configuration via environment variable `ESTIMATOR_INPUT_JSON_PATH`
for the path to the JSON config file or `ESTIMATOR_INPUT_JSON_ENV_VAR`
for the encoded config string to parse.
"""

# Standard
from pathlib import Path
import base64
import json
import logging
import os
import pickle
import subprocess
import sys
import traceback

# First Party
from fm_training_estimator.config.arguments import EstimateInput, MemoryEstimate
from fm_training_estimator.sdk import estimate_memory, estimate_time, estimate_tokens

logging.basicConfig(level=logging.INFO)


def main():
    ##########
    #
    # Parse arguments
    #
    ##########
    try:
        input_dict = get_input_dict()
        logging.info("estimator launch parsed input json: %s", input_dict)
        if not input_dict:
            raise ValueError(
                "Must set environment variable 'ESTIMATOR_INPUT_JSON_PATH'\
            or 'ESTIMATOR_INPUT_JSON_ENV_VAR'."
            )

    except FileNotFoundError as e:
        logging.error(traceback.format_exc())
        sys.exit(1)
    except (TypeError, ValueError, EnvironmentError) as e:
        logging.error(traceback.format_exc())
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        logging.error(traceback.format_exc())
        sys.exit(1)

    ##########
    #
    # Run the estimator
    #
    ##########
    model_path = os.getenv("ESTIMATOR_MODEL_PATH")
    estimator_input = EstimateInput.from_dict(input_dict)

    out_path = os.getenv("ESTIMATOR_OUTPUT_PATH", "estimator_output")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    out_content = "Input parsed for this estimate: " + str(estimator_input) + "\n\n"

    ############ Memory ############
    out_content += "Estimating Memory:....\n"

    memory_output = estimate_memory(estimator_input)
    f = open(os.path.join(out_path, "memory_theory.json"), "w")
    f.write(json.dumps(memory_output.__dict__))
    f.close()

    out_content += "With only theory: " + str(memory_output) + "\n"
    if model_path:
        memory_output = estimate_memory(estimator_input, model_path)
        out_content += "With reg model: " + str(memory_output) + "\n"
        f = open(os.path.join(out_path, "memory_hybrid.json"), "w")
        f.write(json.dumps(memory_output.__dict__))
        f.close()

    ############ Time ############
    out_content += "\n" * 3
    out_content += "Estimating Time:....\n"

    time_output = estimate_time(estimator_input)
    f = open(os.path.join(out_path, "time_theory.json"), "w")
    f.write(json.dumps(time_output.__dict__))
    f.close()

    out_content += "With only theory: " + str(time_output) + "\n"
    if model_path:
        time_output = estimate_time(estimator_input, model_path)
        out_content += "With reg model: " + str(time_output) + "\n"
        f = open(os.path.join(out_path, "time_hybrid.json"), "w")
        f.write(json.dumps(time_output.__dict__))
        f.close()

    ############ Tps ############
    out_content += "\n" * 3
    out_content += "Estimating tps:....\n"

    tps_output = estimate_tokens(estimator_input)
    f = open(os.path.join(out_path, "tps_theory.json"), "w")
    f.write(json.dumps(tps_output.__dict__))
    f.close()

    out_content += "With only theory: " + str(tps_output) + "\n"
    if model_path:
        tps_output = estimate_tokens(estimator_input, model_path)
        out_content += "With reg model: " + str(tps_output) + "\n"
        f = open(os.path.join(out_path, "tps_hybrid.json"), "w")
        f.write(json.dumps(tps_output.__dict__))
        f.close()

    print(out_content)

    f = open(os.path.join(out_path, "output.txt"), "w")
    f.write(out_content)
    f.close()
    return 0


def get_input_dict():
    """Parses JSON configuration if provided via environment variables
    ESTIMATOR_INPUT_JSON_ENV_VAR or ESTIMATOR_INPUT_JSON_PATH.

    ESTIMATOR_INPUT_JSON_ENV_VAR is the base64 encoded JSON.
    ESTIMATOR_INPUT_JSON_PATH is the path to the JSON config file.

    Returns: dict or {}
    """
    json_env_var = os.getenv("ESTIMATOR_INPUT_JSON_ENV_VAR")
    json_path = os.getenv("ESTIMATOR_INPUT_JSON_PATH")

    # accepts either path to JSON file or encoded string config
    # env var takes precedent
    input_dict = {}
    if json_env_var:
        input_dict = txt_to_obj(json_env_var)
    elif json_path:
        with open(json_path, "r", encoding="utf-8") as f:
            input_dict = json.load(f)

    return input_dict


def txt_to_obj(txt):
    """Given encoded byte string, converts to base64 decoded dict.

    Args:
        txt: str
    Returns: dict[str, Any]
    """
    base64_bytes = txt.encode("ascii")
    message_bytes = base64.b64decode(base64_bytes)
    try:
        # If the bytes represent JSON string
        return json.loads(message_bytes)
    except UnicodeDecodeError:
        # Otherwise the bytes are a pickled python dictionary
        return pickle.loads(message_bytes)


if __name__ == "__main__":
    main()
