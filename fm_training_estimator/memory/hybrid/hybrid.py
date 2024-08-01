# Standard
import logging

# Local
from ...config import FMArguments, HFTrainingArguments, InfraArguments, is_fsdp
from ...regressor import LookupRegressor, XGBoostRegressor
from ...utils import extract_model_features
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
        use_model_features=False,
    ):

        logging.info("Hybrid Estimator: Initializing")

        self.fm = fm_args
        self.ta = train_args
        self.ia = infra_args

        self.use_model_features = use_model_features

        self.full_est = FullParameterTuningEstimator(fm_args, train_args)

        if not is_fsdp(train_args):
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

        if self.ia.numGpusPerPod is not None:
            self.fsdp_est.set_number_of_gpus(self.ia.numGpusPerPod)

        # Lookup based estimator
        if lookup_data_path is not None:
            self.lookup_est = LookupRegressor(lookup_data_path)
        else:
            self.lookup_est = None

        # Model based estimator
        if model_path is not None:
            self.reg_est = XGBoostRegressor(model_path)
        else:
            self.reg_est = None

    def lookup_mem(self):
        lookup_query = {
            "model_name": self.fm.base_model_path,
            "number_gpus": self.ia.numGpusPerPod,
            "batch_size": self.ta.per_device_train_batch_size,
            "seq_len": self.fm.block_size,
        }

        if self.use_model_features:
            model_name = lookup_query.pop("model_name")
            lookup_query = lookup_query | extract_model_features(model_name)

        res = self.lookup_est.run(lookup_query)

        if res.empty:
            return None

        return res["memory"][0:1].item()

    def calculate_activation_memory(self):
        if not self.fsdp_enabled:
            return self.full_est.calculate_activation_memory()

        if self.reg_est is None:
            logging.info("Hybrid: Skipping Regression")
            return self.fsdp_est.calculate_activation_memory()

        logging.info("Hybrid: Attempting Regression")

        params = [
            self.fm.base_model_path,
            self.ia.numGpusPerPod,
            self.ta.per_device_train_batch_size,
            self.fm.block_size,
        ]

        if self.use_model_features:
            model_name = params[0]
            params = params[1:] + extract_model_features(model_name, fmt="list")

        res = self.reg_est.run(params)

        # activation memory are 3rd entry in the list
        return res[0][2]

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
            logging.info("Hybrid: attempting lookup")
            lookup_mem = self.lookup_mem()
            if lookup_mem is not None:
                logging.info("Lookup: match found")
                return lookup_mem

        logging.info("Hybrid: lookup failed")

        size = (
            self.calculate_activation_memory()
            + self.fsdp_est.calculate_gradient_memory()
            + self.fsdp_est.calculate_model_memory()
            + self.fsdp_est.calculate_optimizer_memory()
        )

        return size
