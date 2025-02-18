# Third Party
from accelerate import init_empty_weights
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM

# Local
from ...config import FMArguments, HFTrainingArguments, PeftLoraConfig
from ...utils import fmt_size, get_size_from_precision, logger
from ..full import FullParameterTuningEstimator


class LoraEstimator(FullParameterTuningEstimator):
    def __init__(
        self,
        fm_args: FMArguments,
        train_args: HFTrainingArguments,
        lora_args: PeftLoraConfig,
    ):
        super().__init__(fm_args, train_args)

        self.train_args = train_args
        self.fm_args = fm_args
        self.lora_args = lora_args

        with init_empty_weights():
            modelc = AutoConfig.from_pretrained(self.fm_args.base_model_path)
            model = AutoModelForCausalLM.from_config(modelc)

        logger.info("Initializing LoraEstimator with lora args %s", self.lora_args)
        self.peft_model = get_peft_model(
            model,
            LoraConfig(
                r=self.lora_args.r,
                lora_alpha=self.lora_args.lora_alpha,
                lora_dropout=self.lora_args.lora_dropout,
                target_modules=self.lora_args.target_modules,
            ),
        )

        self.num_of_trainable_params = self.peft_model.num_parameters(
            only_trainable=True
        )
        self.num_of_model_params = self.peft_model.num_parameters()

        self.precision = self._get_precision()

    def calculate_activation_memory(self, readable=False):
        # tensors created during forward pass that are needed for gradient computation
        # outputs have to be stored which will be used during backward pass
        peft_model_state = self.peft_model.state_dict()
        lora_a = []
        lora_b = []
        lora_dropout = []
        either_q_k_v_present = False
        for k in peft_model_state:
            if "lora_A" in k:
                lora_a.append(peft_model_state[k])
            if "lora_B" in k:
                lora_b.append(peft_model_state[k])
            if "lora_dropout" in k:
                lora_dropout.append(peft_model_state[k])
            if "self_attn" in k:
                either_q_k_v_present = True
        # for each trainable linear layer
        # input_features * batch_size * seq_length elements needed for each layer
        lora_a_size = 0
        lora_b_size = 0
        lora_dropout_size = 0
        # single shared input for Q K V matrices
        input_size = 0
        if either_q_k_v_present:
            input_size = (
                self.h * self.b * self.s * get_size_from_precision(self.precision)
            )
        for lora_a_i in lora_a:
            lora_a_size += (
                lora_a_i.size()[1]
                * self.b
                * self.s
                * get_size_from_precision(self.precision)
            )
        for lora_b_i in lora_b:
            lora_b_size += (
                lora_b_i.size()[1]
                * self.b
                * self.s
                * get_size_from_precision(self.precision)
            )
        for lora_dropout_i in lora_dropout:
            lora_dropout_size += lora_dropout_i.size()[1] * self.b * self.s
        # ignored 2 layer normalization layers and softmax
        size = input_size + lora_a_size + lora_b_size + lora_dropout_size
        if readable:
            return fmt_size(size)
        return size
