import json

# Third Party
import fire
import gradio as gr
import uvicorn

# Local
from .api import api
from .core import run

# control variables
conf_model_path = None
conf_data_path = None

# list of white-listed models
model_list = []


def to_config(
    base_model_path,
    block_size,
    batch_size,
    model_precision,
    grad_checkpoint,
    num_gpus,
    gpu_model,
    gpu_mem,
    technique,
    token_est_approach,
    dataset,
    dataset_field,
    dataset_split,
    dataset_config,
    dataset_config_file,
):
    config = {
        "base_model_path": base_model_path,
        "block_size": block_size,
        "per_device_train_batch_size": batch_size,
        "torch_dtype": model_precision,
        "gradient_checkpointing": grad_checkpoint,
        "numGpusPerPod": num_gpus,
        "gpuModel": gpu_model,
        "gpu_memory_in_gb": gpu_mem,
        "technique": technique,
        "dataset": dataset,
        "dataset_text_field": dataset_field,
        "dataset_split": dataset_split,
        "dataset_config_name": dataset_config,
        "dataset_config_file": dataset_config_file,
    }

    # can be 0, for auto, or > 1 for manual
    if num_gpus != 1:
        config["fsdp"] = "full_shard"

    match token_est_approach:
        case "disabled":
            config["te_approach"] = -1
        case "0":
            config["te_approach"] = 0
        case "2":
            config["te_approach"] = 2

    return config


def update_conf(*args):
    config = to_config(*args)
    return config


def estimate(*args):
    prev_out = args[-1]
    prev_conf = args[-2]
    args = args[:-2]

    config = to_config(*args)
    # conf_store.value = config

    output = run(config, conf_data_path, conf_model_path)
    # remove out the "_og" fields
    rem_keys = list(filter(lambda x: x.endswith("_og"), output.keys()))
    for k in rem_keys:
        del output[k]

    return [
        config,
        output,
        prev_conf,
        prev_out,
    ]


def web(
    model_whitelist=None,
    data_path=None,
    model_path=None,
    port=3000,
    enable_api=False,
):
    """
    model_whitelist: path to a text file, with a list of models to show in the dropdown
    port: Port to start the webserver on
    data_path: Path to data file for lookup
    model_path: Path to model file with regression model
    enable_api: whether to enable the api as a part of this ui
    """

    global model_list
    if model_whitelist is not None:
        with open(model_whitelist, "r") as wl:
            model_list = wl.read().splitlines()

    global conf_data_path, conf_model_path
    conf_data_path = data_path
    conf_model_path = model_path

    with gr.Blocks(title="fm-training-estimator") as demo:
        with gr.Row():
            with gr.Column():
                if model_list == []:
                    base_model_path = gr.Textbox(
                        label="Model",
                        info=("Model name/url in HF format"),
                        value="ibm-granite/granite-7b-base",
                    )
                else:
                    base_model_path = gr.Dropdown(model_list, label="Model")

                block_size = gr.Slider(
                    label="Sequence length",
                    info=(
                        "Number of tokens in a sample sequence. If the chosen sequence length is "
                        "greater than model's max seq length then its set to model's max_seq_length"
                    ),
                    minimum=512,
                    maximum=32 * 1024,
                    value=1024,
                    step=512,
                )

                batch_size = gr.Slider(
                    label="Batch size (per device)",
                    info="number of samples per batch per device",
                    minimum=1,
                    maximum=512,
                    value=1,
                    step=1,
                )

                model_precision = gr.Dropdown(
                    label="Model precision",
                    choices=["float16", "float32", "bfloat16"],
                    value="bfloat16",
                )

                grad_checkpoint = gr.Checkbox(
                    label="Gradient Checkpointing Enabled?",
                    info="(Experimental Feature)",
                    value=False,
                )

                num_gpus = gr.Slider(
                    label="Number of GPUs",
                    info="set to 0 to auto-detect",
                    minimum=0,
                    maximum=32,
                    value=0,
                    step=1,
                )

                gpu_model = gr.Dropdown(
                    label="GPU Model", choices=["A100", "H100", "L40S"], value="A100"
                )
                gpu_mem = gr.Dropdown(
                    label="GPU Memory (per GPU)", choices=[48, 40, 80], value=80
                )

                def update_gpu_memory(gpu_model):
                    if gpu_model == "L40S":
                        return 48
                    else:
                        return 80

                gpu_model.change(update_gpu_memory, gpu_model, gpu_mem)

                technique = gr.Dropdown(
                    ["full", "lora", "qlora"],
                    value="full",
                    label="Technique",
                    info="All approaches use FSDP to scale. Other techniques will be added later!",
                )

                with gr.Row():
                    with gr.Column():

                        token_est_approach = gr.Dropdown(
                            ["disabled", "0", "2"],
                            value="disabled",
                            label="Token Estimation Approach",
                        )
                            
                        
                        dataset = gr.Textbox(
                            label="Dataset",
                            info="Name/path of dataset in HF datasets format",
                            visible=False,
                        )

                        dataset_field = gr.Textbox(
                            label="Dataset Field",
                            value="text",
                            info="Field to use during training",
                            visible=False,
                        )

                        dataset_split = gr.Textbox(
                            label="Dataset Split", 
                            value="test", 
                            visible=False
                        )

                        dataset_config = gr.Textbox(
                            label="Dataset Config", 
                            visible=False
                        )

                        dataset_config_file = gr.File(
                            label="Upload dataset configuration file", 
                            visible=False
                        )

                        # Function to toggle visibility
                        def toggle_visibility(value):
                            if value == "0":
                                return {
                                    dataset: gr.update(visible=True),
                                    dataset_field: gr.update(visible=True),
                                    dataset_split: gr.update(visible=True),
                                    dataset_config: gr.update(visible=True),  # visible only when value == 1
                                    dataset_config_file: gr.update(visible=False),
                                }
                            elif value == "2":
                                return {
                                    dataset: gr.update(visible=False),
                                    dataset_field: gr.update(visible=True),
                                    dataset_split: gr.update(visible=False),
                                    dataset_config: gr.update(visible=False),  # visible only when value == 1
                                    dataset_config_file: gr.update(visible=True),
                                }
                            else:
                                return {
                                    dataset: gr.update(visible=False),
                                    dataset_field: gr.update(visible=False),
                                    dataset_split: gr.update(visible=False),
                                    dataset_config: gr.update(visible=False),  # visible only when value == 1
                                    dataset_config_file: gr.update(visible=False),
                                }
                    

                        # Bind the visibility toggle to the dropdown
                        token_est_approach.change(
                            toggle_visibility,
                            token_est_approach,
                            [dataset, dataset_field, dataset_split, dataset_config, dataset_config_file],
                        )

                submit_btn = gr.Button("Submit")
                to_conf_btn = gr.Button("Gen Config")

                inputs = [
                    base_model_path,
                    block_size,
                    batch_size,
                    model_precision,
                    grad_checkpoint,
                    num_gpus,
                    gpu_model,
                    gpu_mem,
                    technique,
                    token_est_approach,
                    dataset,
                    dataset_field,
                    dataset_split,
                    dataset_config,
                    dataset_config_file
                ]
            with gr.Column():
                with gr.Accordion("Estimation"):
                    outputs = gr.JSON(label="Predicted Resources")
                with gr.Accordion("Configuration", open=False):
                    conf = gr.JSON(label="Conf")

                with gr.Accordion("Previous Estimation"):
                    prev_outputs = gr.JSON(label="Prev Predicted Resources")
                with gr.Accordion("Previous Configuration", open=False):
                    prev_conf = gr.JSON(label="Prev Conf")

        submit_btn.click(
            estimate,
            inputs=inputs + [conf, outputs],
            outputs=[conf, outputs, prev_conf, prev_outputs],
        )
        to_conf_btn.click(update_conf, inputs=inputs, outputs=conf)

    if enable_api:
        app = api(data_path, model_path)
        gr.mount_gradio_app(app, demo, path="/")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        demo.queue()
        demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    fire.Fire(web)
