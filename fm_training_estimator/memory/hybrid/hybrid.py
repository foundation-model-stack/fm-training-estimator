# Local
from ...config import FMArguments, HFTrainingArguments, InfraArguments, is_fsdp
from ...data import format_query
from ...regressor import LookupRegressor, GetRegressor
from ...utils import logger
from ..fsdp import FSDPEstimator
from ..full import FullParameterTuningEstimator


class HybridEstimator:
    def __init__(
        self,
        fm_args: FMArguments,
        train_args: HFTrainingArguments,
        infra_args: InfraArguments,
        lookup_data_path,
        model_path,
    ):

        logger.info("Memory Hybrid: Initializing")

        self.fm = fm_args
        self.ta = train_args
        self.ia = infra_args

        # if fsdp param is not set, set it to default
        if self.ia.numGpusPerPod != 1:
            if self.ta.fsdp == []:
                self.ta.fsdp = ["full_shard"]

        self.full_est = FullParameterTuningEstimator(fm_args, train_args)

        if not is_fsdp(self.ta):
            self.fsdp_enabled = False
            return

        # FSDP related logic
        self.fsdp_enabled = True
        self.fsdp_est = FSDPEstimator(
            fm_args,
            train_args,
            self.full_est,
            infra_args.gpu_memory_in_gb * 1024 * 1024 * 1024,
        )

        self.fsdp_est.set_number_of_gpus(self.ia.numGpusPerPod)

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

        # auto-discover?
        if self.ia.numGpusPerPod == 0:
            self.auto_discover_num_gpus()

    def auto_discover_num_gpus(self):
        """Discover the number of gpus needed - by guess and emperical validation."""
        logger.info("Memory Hybrid - Attempting auto discovery of num gpus...")

        guess = self.fsdp_est.estimate_number_of_gpus()
        trials = 10

        while trials > 0:
            self.fsdp_est.set_number_of_gpus(guess)
            mem = self.get_total_mem_estimate()

            # acceptable memory configuration found
            if mem < self.ia.gpu_memory_in_gb * 1024**3:
                logger.debug(
                    "Memory Hybrid - finalized num of gpus to: {}".format(guess)
                )
                return

            guess += 1
            trials -= 1

        logger.warning("Memory Hybrid - No suitable num gpus found!")
        self.fsdp_est.set_number_of_gpus(-1)

    def lookup_mem(self):
        lookup_query = {
            "model_name": self.fm.base_model_path,
            "number_gpus": self.fsdp_est.num_gpus,
            "batch_size": self.ta.per_device_train_batch_size,
            "seq_len": self.fm.block_size,
            "gpu_model": self.ia.gpuModel,
            "method": self.fm.technique,
        }

        lookup_query = format_query(lookup_query, self.lookup_est.get_data_format())

        res = self.lookup_est.run(lookup_query)

        if res.empty:
            return None

        return res["memory"][0:1].item()

    def calculate_activation_memory(self):
        if not self.fsdp_enabled:
            return self.full_est.calculate_activation_memory()

        if self.reg_est is None:
            logger.debug("Memory Hybrid - Skipping Regression")
            return self.fsdp_est.calculate_activation_memory()

        logger.debug("Memory Hybrid - Attempting Regression")

        lookup_query = {
            "model_name": self.fm.base_model_path,
            "number_gpus": self.fsdp_est.num_gpus,
            "batch_size": self.ta.per_device_train_batch_size,
            "seq_len": self.fm.block_size,
            "gpu_model": self.ia.gpuModel,
            "method": self.fm.technique,
        }

        params = format_query(
            lookup_query, self.reg_est.get_data_format(), only_values=True
        )

        res = self.reg_est.run(params)

        # activation memory are 3rd entry in the list
        act = res[0][2]

        logger.info(
            "Memory Hybrid - Activation, from regression: {}, from theory: {}".format(
                act, self.fsdp_est.calculate_activation_memory()
            )
        )

        return act

    def calculate_gradient_memory(self):
        if not self.fsdp_enabled:
            return self.full_est.calculate_gradient_memory()

        return self.fsdp_est.calculate_gradient_memory()

    def calculate_model_memory(self):
        if not self.fsdp_enabled:
            return self.full_est.calculate_model_memory()

        return self.fsdp_est.calculate_model_memory()

    def calculate_optimizer_memory(self):
        if not self.fsdp_enabled:
            return self.full_est.calculate_optimizer_memory()

        return self.fsdp_est.calculate_optimizer_memory()

    def get_total_mem_estimate(self):
        if not self.fsdp_enabled:
            return self.full_est.get_total_mem_estimate()

        # simple lookup
        if self.lookup_est is not None:
            logger.debug("Memory Hybrid - attempting lookup")
            lookup_mem = self.lookup_mem()
            if lookup_mem is not None:
                logger.debug("Memory Hybrid - match found")
                return lookup_mem

        logger.info("Memory Hybrid - lookup failed")

        size = (
            self.calculate_activation_memory()
            + self.fsdp_est.calculate_gradient_memory()
            + self.fsdp_est.calculate_model_memory()
            + self.fsdp_est.calculate_optimizer_memory()
        )

        return size
