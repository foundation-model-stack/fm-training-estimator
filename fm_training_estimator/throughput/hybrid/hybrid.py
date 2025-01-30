# Local
from ...config import FMArguments, HFTrainingArguments, InfraArguments
from ...data import format_query
from ...regressor import LookupRegressor, GetRegressor
from ...utils import logger


class HybridSpeedEstimator:
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
        self.lookup_est = None
        self.reg_est = None

        # Lookup based estimator
        if lookup_data_path is not None:
            self.lookup_est = LookupRegressor(lookup_data_path)

        # Model based estimator
        if model_path is not None:
            self.reg_est = GetRegressor(model_path)

        if lookup_data_path is None and model_path is None:
            raise RuntimeError("HybridSpeedEstimator not properly initialized")

    def check_lookup(self, seqlen):
        lookup_query = {
            "model_name": self.fm.base_model_path,
            "number_gpus": self.ia.numGpusPerPod,
            "batch_size": self.ta.per_device_train_batch_size,
            "seq_len": seqlen,
            "gpu_model": self.ia.gpuModel,
            "method": self.fm.technique,
        }

        lookup_query = format_query(lookup_query, self.lookup_est.get_data_format())

        res = self.lookup_est.run(lookup_query)

        if res.empty:
            return None

        logger.debug(f"Throughput Hybrid - Lookup result: {res}")
        return res[0:1]["tokens_per_second"].item()

    def get_tps(self, seqlen=None):
        if seqlen is None:
            seqlen = self.fm.block_size

        res = None

        # attempt lookup
        if self.lookup_est is not None:
            res = self.check_lookup(seqlen)
            if res is not None:
                return res
        if self.reg_est is None:
            return res

        # attempt reg approach
        lookup_query = {
            "model_name": self.fm.base_model_path,
            "number_gpus": self.ia.numGpusPerPod,
            "batch_size": self.ta.per_device_train_batch_size,
            "seq_len": int(seqlen),
            "gpu_model": self.ia.gpuModel,
            "method": self.fm.technique,
        }
        params = format_query(
            lookup_query, self.reg_est.get_data_format(), only_values=True
        )

        res = self.reg_est.run(params)

        # tps is 1st entry in the list
        return res[0][0]
