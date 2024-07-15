# Third Party
import fire
import gradio as gr

# Local
from .core import run

# list of white-listed models
model_list = []
# stores the current configuration specified in the ui as json
# this json can be used as-is with the cli
conf_store = gr.State(value={})


def to_config(
    base_model_path,
    block_size,
    batch_size,
    model_precision,
    num_gpus,
    gpu_mem,
    technique,
):
    config = {
        "base_model_path": base_model_path,
        "block_size": block_size,
        "per_device_train_batch_size": batch_size,
        "torch_dtype": model_precision,
        "numGpusPerPod": num_gpus,
        "gpu_memory_in_gb": gpu_mem,
    }

    if config["numGpusPerPod"] == 0:
        del config["numGpusPerPod"]

    if technique == "fsdp":
        config["fsdp"] = "full_shard"

    conf_store.value = config
    return config


def update_conf(*args):
    config = to_config(*args)
    return config


def estimate(*args):
    config = to_config(*args)
    conf_store.value = config

    return [config, run(config)]


def web(model_whitelist=None, port=3000):
    """
    model_whitelist: path to a text file, with a list of models to show in the dropdown
    port: Port to start the webserver on
    """

    global model_list
    if model_whitelist is not None:
        with open(model_whitelist, "r") as wl:
            model_list = wl.read().splitlines()

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

                submit_btn = gr.Button("Submit")
                to_conf_btn = gr.Button("Gen Config")

                conf = gr.JSON(label="Conf")

                inputs = [
                    base_model_path,
                    block_size,
                    batch_size,
                    model_precision,
                    num_gpus,
                    gpu_mem,
                    technique,
                ]
            with gr.Column():
                outputs = gr.JSON(label="Predicted Memory")

        submit_btn.click(estimate, inputs=inputs, outputs=[conf, outputs])
        to_conf_btn.click(update_conf, inputs=inputs, outputs=conf)

    demo.queue()
    demo.launch(server_port=port)


if __name__ == "__main__":
    fire.Fire(web)
