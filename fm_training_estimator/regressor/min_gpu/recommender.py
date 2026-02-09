import logging
from numpy import rec
from orchestrator.schema.point import SpacePoint
from orchestrator.schema.reference import (
    ExperimentReference,
)
from orchestrator.modules.actuators.registry import ActuatorRegistry

from autoconf.min_gpu_recommender import min_gpu_recommender, avoid_oom_recommender
from autoconf.utils import config_mapper

logger = logging.getLogger(__name__)


class MinGpuRecommenderCaller(object):

    # def __init__(self):
    # Ideally we should ask the API to load and initialise the model here.
    # Right now our AutoConf API has only one call which
    # loads and does an inference on the recommender model

    #def normalize_config_dict(self, X:dict) -> dict:

    def run(self, X: dict, y: str="avoid_oom") -> dict:
        """
        Run the GPU recommender to determine optimal compute configuration.
        
        This method takes a training configuration and recommends the number of workers
        and GPUs per worker needed to either minimize GPU usage or avoid OOM errors.
        
        Args:
            X (dict): Training configuration dictionary containing:
                - model_name (str): Path or identifier of the model
                - method (str): Training method (e.g., "full", "lora")
                - gpu_model (str): GPU model identifier (e.g., "NVIDIA-A100-SXM4-80GB")
                - tokens_per_sample (int): Maximum sequence length/tokens per sample
                - per_device_train_batch_size or batch_size (int): Batch size per device or the total batch size
                
            y (str, optional): Recommendation strategy. Defaults to "avoid_oom".
                - "avoid_oom": Recommend configuration to avoid out-of-memory errors
                - "min_gpu": Recommend minimum GPU configuration needed
                
        Returns:
            dict: Recommendation result containing:
                - workers (int): Recommended number of worker nodes (-1 if cannot recommend)
                - gpus_per_worker (int): Recommended GPUs per worker (-1 if cannot recommend)
                
        Raises:
            ValueError: If an unknown recommendation type is provided for parameter y
            
        Note:
            If the recommender cannot make a recommendation or encounters an error,
            both workers and gpus_per_worker will be set to -1.
        """
        #X = normalize_config_dict(X)
        logger.debug(f'Received input {X} for recommendation')
        
        try:
            #Map model name to valid config values
            config = config_mapper.map_to_valid(X)
            #e = SpacePoint.model_validate({"entity":config}).to_entity()
            #logger.debug(f'Running Entity {e} against {self.experiment}')
            match(y):
                case "min_gpu":
                    rec_result = min_gpu_recommender(**config)
                    logger.debug(f'Received recommendation {rec_result}')
                case "avoid_oom":
                    # Use min_gpu_recommender for avoid_oom as well
                    # The difference is in how the result is interpreted/used by the caller
                    rec_result = avoid_oom_recommender(**config)
                    logger.debug(f'Received recommendation {rec_result}')
                case _:
                    raise ValueError(f"Unknown recommendation type {y}")

            result = {}
            if not rec_result.get("can_recommend",False):
                logger.debug("Recommender could not recommend")
                result["workers"] = -1
                result["gpus_per_worker"] = -1
            else:
                result["workers"] = rec_result.get("workers", -1)
                result["gpus_per_worker"] = rec_result.get("gpus", -1)
            logger.warning(result)
  
        except Exception as e:
            logger.warning(f'Caught exception {e}')
            result = {}
            result["workers"] = -1
            result["gpus_per_worker"] = -1
        finally:
            logger.info("Goodbye")

        return result
