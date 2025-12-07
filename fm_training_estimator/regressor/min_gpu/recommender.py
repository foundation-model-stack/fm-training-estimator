import logging
from orchestrator.schema.point import SpacePoint
from orchestrator.schema.reference import (
    ExperimentReference,
)
from orchestrator.modules.actuators.registry import ActuatorRegistry

from autoconf.min_gpu_recommender import (
    min_gpu_recommender,
)
from autoconf.utils import config_mapper

logger = logging.getLogger(__name__)


class MinGpuRecommenderCaller(object):

    # def __init__(self):
    # Ideally we should ask the API to load and initialise the model here.
    # Right now our AutoConf API has only one call which
    # loads and does an inference on the recommender model

    def __init__(self):
        self._experiment = ActuatorRegistry().experimentForReference(
                ExperimentReference(
                    actuatorIdentifier="custom_experiments",
                    experimentIdentifier="min_gpu_recommender",
                )
            )

    @property
    def experiment(self)->ExperimentReference:
        return self._experiment

    @experiment.setter
    def set_experiment(self, ref: ExperimentReference):
        if self._experiment == None:
            self._experiment = ref

    #def normalize_config_dict(self, X:dict) -> dict:

    def run(self, X: dict, y: str) -> dict:

        #X = normalize_config_dict(X)
        logger.debug(f'Received input {X} for recommendation')
        
        try:
            #Map model name to valid config values
            config = config_mapper.map_to_valid(X)
            e = SpacePoint.model_validate({"entity":config}).to_entity()
            logger.debug(f'Running Entity {e} against {self.experiment}')
            rec_result = min_gpu_recommender(**config)
            logger.debug(f'Received recommendation {rec_result}')
            result = {}
            for k,v in rec_result.items():
                if k == "can_recommend":
                    if not v:
                        logger.debug(f"Recommender could not recommend")
                        result["workers"] = -1
                        result["gpus_per_worker"] = -1
                        break
            
                if  k == "workers":
                    result["workers"] = v
                if k == "gpus":
                    result["gpus_per_worker"] = v
        
            logger.warning(result)
    
        except Exception as e:
            logger.warning(f'Caught exception {e}')
            result = {}
            result["workers"] = -1
            result["gpus_per_worker"] = -1
        finally:
            logger.info("Goodbye")

        return result
