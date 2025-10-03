from ado_actuators.min_gpu_recommender import api


class MinGpuRecommender(object):

    #def __init__(self):
        #Ideally we should ask the API to load and initialise the model here. 
        #Right now our AutoConf API has only one call which
        #loads and does an inference on the recommender model


    def run(self, 
            model_name:str, 
            method: str,
            gpu_model: str, 
            tokens_per_sample: int, 
            batch_size: int) -> (int,int):
        
        try:
            result: api.GPUsAndWorkers = api.recommend_min_gpus_and_workers(
                model_name=model_name,
                method=method,
                gpu_model=gpu_model,
                number_gpus=1,
                tokens_per_sample=tokens_per_sample,
                batch_size=batch_size,
            )
        except api.NoRecommendationError:
            print("No recommendation provided")
            return (-1, -1)

        return (result.workers, result.gpus)