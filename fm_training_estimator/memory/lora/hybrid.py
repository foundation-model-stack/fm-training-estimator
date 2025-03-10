# Local
from ...config import FMArguments, HFTrainingArguments, InfraArguments, PeftLoraConfig
from ...data import format_query
from ...regressor import LookupRegressor, GetRegressor
from ...utils import logger
from .lora import LoraEstimator


class HybridLoraEstimator:
    def __init__(
        self,
        fm_args: FMArguments,
        train_args: HFTrainingArguments,
        infra_args: InfraArguments,
        lora_args: PeftLoraConfig,
        lookup_data_path,
        model_path,
    ):

        logger.info("Memory Lora Hybrid - Initializing")

        self.fm = fm_args
        self.ta = train_args
        self.ia = infra_args

        self.lora_est = LoraEstimator(fm_args, train_args, lora_args)

        # Lookup based estimator
        if lookup_data_path is not None:
            self.lookup_est = LookupRegressor(lookup_data_path)
        else:
            self.lookup_est = None

        # Model based estimator
        if model_path is not None:
            self.reg_est = GetRegressor(model_path)
        else:
            self.reg_est = None

        if self.ia.numGpusPerPod == 0:
            # discover number of gpus
            self.auto_discover_num_gpus()
        else:
            self.num_gpus = self.ia.numGpusPerPod

    def auto_discover_num_gpus(self):
        num = self.lora_est.calculate_model_memory() / (
            self.ia.gpu_memory_in_gb * 1024**3
        )
        self.num_gpus = int(num) if num > 1 else 1

        trials = 10
        while trials > 0:
            mem = self.get_total_mem_estimate()
            if mem < self.ia.gpu_memory_in_gb * 1024**3:
                logger.info(
                    "Memory Lora Hybrid - Discovered num gpus: {0}".format(
                        self.num_gpus
                    )
                )
                return

            trials -= 1
            self.num_gpus += 1

        logger.warning("Memory Lora Hybrid - No suitable num gpus found!")

    def calculate_model_memory(self):
        return self.lora_est.calculate_model_memory() / self.num_gpus

    def calculate_gradient_memory(self):
        return self.lora_est.calculate_gradient_memory() / self.num_gpus

    def calculate_optimizer_memory(self):
        return self.lora_est.calculate_optimizer_memory() / self.num_gpus

    def calculate_activation_memory(self):
        return self.lora_est.calculate_activation_memory() / self.num_gpus

    def get_total_mem_estimate(self):

        lookup_query_base = {
            "model_name": self.fm.base_model_path,
            "number_gpus": self.num_gpus,
            "batch_size": self.ta.per_device_train_batch_size,
            "seq_len": self.fm.block_size,
            "gpu_model": self.ia.gpuModel,
            "method": self.fm.technique,
        }

        if self.lookup_est is not None:
            logger.debug("Memory Lora Hybrid - Attempting lookup")
            lookup_query = format_query(
                lookup_query_base, self.lookup_est.get_data_format()
            )
            logger.debug("Memory Lora Hybrid - Lookup query for lookup_est is: %s", lookup_query)
            res = self.lookup_est.run(lookup_query)
            if res.empty:
                lookup_mem = None
                logger.debug(
                    "Memory Lora Hybrid - No match was found by lookup, trying reg_est"
                )
            else:
                lookup_mem = res["memory"][0:1].item()
            if lookup_mem is not None:
                logger.info("Memory Lora Hybrid - Lookup: match found")
                return lookup_mem

        if self.reg_est is not None:
            params = format_query(
                lookup_query_base, self.reg_est.get_data_format(), only_values=True
            )
            logger.debug("Memory Lora Hybrid - Lookup query for reg_est is: %s", params)
            act = self.reg_est.run(params, "memory")
            logger.debug("Memory Lora Hybrid - Lookup query result for reg_est is: %s", act)

            return act

        # If we reach here, we are falling back on theory
        size = (
            self.calculate_activation_memory()
            + self.lora_est.calculate_gradient_memory()
            + self.lora_est.calculate_model_memory()
            + self.lora_est.calculate_optimizer_memory()
        )

        return size
