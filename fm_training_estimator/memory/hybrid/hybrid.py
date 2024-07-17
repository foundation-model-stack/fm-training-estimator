# Local
from ...config import FMArguments, HFTrainingArguments, InfraArguments, is_fsdp
from ...regressor import LookupRegressor, XGBoostRegressor
from ...utils import fmt_size
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

        self.fm = fm_args
        self.ta = train_args
        self.ia = infra_args

        self.full_est = FullParameterTuningEstimator(fm_args, train_args)

        if not is_fsdp(train_args):
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
        self.lookup_est = LookupRegressor(lookup_data_path)
        # Model based estimator
        self.reg_est = XGBoostRegressor(model_path)

    def lookup_mem(self):
        res = self.lookup_est.run(
            {
                "model_name": self.fm.base_model_path,
                "number_gpus": self.ia.numGpusPerPod,
                "batch_size": self.ta.per_device_train_batch_size,
                "seq_len": self.fm.block_size,
            }
        )

        if res.empty:
            return None

        return res["memory"][0]

    def calculate_activation_memory(self):
        if not self.fsdp_enabled:
            return self.full_est.calculate_activation_memory()

        params = [
            self.fm.base_model_path,
            self.ia.numGpusPerPod,
            self.ta.per_device_train_batch_size,
            self.fm.block_size,
        ]
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

    def get_total_mem_estimate(self, readable: bool = False):
        if not self.fsdp_enabled:
            return self.full_est.get_total_mem_estimate(readable)

        # simple lookup
        lookup_mem = self.lookup_mem()
        if lookup_mem is not None:
            if readable:
                return fmt_size(lookup_mem)
            return lookup_mem

        size = (
            self.calculate_activation_memory()
            + self.fsdp_est.calculate_gradient_memory()
            + self.fsdp_est.calculate_model_memory()
            + self.fsdp_est.calculate_optimizer_memory()
        )

        if readable:
            return fmt_size(size)
        return size
