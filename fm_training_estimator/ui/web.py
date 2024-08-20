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
conf_umf = True

# list of white-listed models
model_list = []


def to_config(
    base_model_path,
    block_size,
    batch_size,
    model_precision,
    grad_checkpoint,
    num_gpus,
    gpu_mem,
    technique,
    token_est_approach,
    dataset,
    dataset_field,
    dataset_split,
    dataset_config,
):
    config = {
        "base_model_path": base_model_path,
        "block_size": block_size,
        "per_device_train_batch_size": batch_size,
        "torch_dtype": model_precision,
        "gradient_checkpointing": grad_checkpoint,
        "numGpusPerPod": num_gpus,
        "gpu_memory_in_gb": gpu_mem,
        "dataset": dataset,
        "dataset_text_field": dataset_field,
        "dataset_split": dataset_split,
        "dataset_config_name": dataset_config,
    }

    if technique == "fsdp":
        config["fsdp"] = "full_shard"

    match token_est_approach:
        case "disabled":
            config["te_approach"] = -1
        case "0":
            config["te_approach"] = 0

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

    return [
        config,
        run(config, conf_data_path, conf_model_path, conf_umf),
        prev_conf,
        prev_out,
    ]


def web(
    model_whitelist=None,
    data_path=None,
    model_path=None,
    port=3000,
    use_model_features=True,
    enable_api=False,
):
    """
    model_whitelist: path to a text file, with a list of models to show in the dropdown
    port: Port to start the webserver on
    data_path: Path to data file for lookup
    model_path: Path to model file with regression model
    use_model_features: whether to use model name or features as the keys for lookup and regression
    enable_api: whether to enable the api as a part of this ui
    """

    global model_list
    if model_whitelist is not None:
        with open(model_whitelist, "r") as wl:
            model_list = wl.read().splitlines()

    global conf_data_path, conf_model_path, conf_umf
    conf_data_path = data_path
    conf_model_path = model_path
    conf_umf = use_model_features

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
                    label="Batch size",
                    info="number of samples per batch",
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

                gpu_mem = gr.Dropdown(
                    label="GPU Memory (per GPU)", choices=[36, 40, 80], value=80
                )

                technique = gr.Dropdown(
                    ["full", "fsdp"],
                    value="full",
                    label="Technique",
                    info="Other techniques will be added later!",
                )

                def update_num_gpus(technique):
                    if technique == "full":
                        return 1

                technique.change(update_num_gpus, technique, num_gpus)

                with gr.Row():
                    with gr.Column():

                        token_est_approach = gr.Dropdown(
                            ["disabled", "0"],
                            value="disabled",
                            label="Token Estimation Approach",
                        )

                        dataset = gr.Textbox(
                            label="Dataset",
                            info="name/path of dataset in HF datasets format",
                        )

                        dataset_field = gr.Textbox(
                            label="Dataset Field",
                            value="text",
                            info="field to use during training",
                        )

                        dataset_split = gr.Textbox(label="Dataset Split", value="test")

                        dataset_config = gr.Textbox(label="Dataset Config")

                submit_btn = gr.Button("Submit")
                to_conf_btn = gr.Button("Gen Config")

                inputs = [
                    base_model_path,
                    block_size,
                    batch_size,
                    model_precision,
                    grad_checkpoint,
                    num_gpus,
                    gpu_mem,
                    technique,
                    token_est_approach,
                    dataset,
                    dataset_field,
                    dataset_split,
                    dataset_config,
                ]
            with gr.Column():
                with gr.Accordion("Configuration"):
                    conf = gr.JSON(label="Conf")
                with gr.Accordion("Estimation"):
                    outputs = gr.JSON(label="Predicted Resources")

                with gr.Accordion("Previous Configuration"):
                    prev_conf = gr.JSON(label="Prev Conf")
                with gr.Accordion("Previous Estimation"):
                    prev_outputs = gr.JSON(label="Prev Predicted Resources")

        submit_btn.click(
            estimate,
            inputs=inputs + [conf, outputs],
            outputs=[conf, outputs, prev_conf, prev_outputs],
        )
        to_conf_btn.click(update_conf, inputs=inputs, outputs=conf)

    if enable_api:
        app = api(data_path, model_path, use_model_features)
        gr.mount_gradio_app(app, demo, path="/")
        uvicorn.run(app, host="0.0.0.0", port=port)
    else:
        demo.queue()
        demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    fire.Fire(web)
