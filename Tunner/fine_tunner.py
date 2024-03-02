import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig
from trl import SFTTrainer

class Llama_Tuner():
    def __init__(self):
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except ImportError:
            pass

    def to_dataset(self, instruction: list, response: list):
        assert len(instruction) == len(response)
        
        dataset = []
        for i in range(len(instruction)):
            item = {}
            item['text'] = "<s>[INST] " + str(instruction[i]) + " [/INST] " + str(response[i]) + " </s>"
            dataset.append(item)
        
        return iter(dataset)
    
    def load_model(self, model_name: str, quant_config: dict, lora_config: dict, train_config: dict):
        self.model_name = model_name
        quant_config = BitsAndBytesConfig(
            load_in_4bit=quant_config['load_in_4bit'],
            bnb_4bit_quant_type=quant_config['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype=quant_config['bnb_4bit_compute_dtype'],
            bnb_4bit_use_double_quant=quant_config['bnb_4bit_use_double_quant']
            )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map={"": 0}
        )

        self.base_model.config.use_cache = False
        self.base_model.config.pretraining_tp = 1

        self.peft_parameters = LoraConfig(
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            r=lora_config['r'],
            bias=lora_config['bias'],
            task_type=lora_config['task_type']
        )

        self.train_params = TrainingArguments(
            output_dir = f"./{train_config['output_dir']}",
            num_train_epochs = train_config['num_train_epochs'],
            per_device_train_batch_size = train_config['per_device_train_batch_size'],
            gradient_accumulation_steps = train_config['gradient_accumulation_steps'],
            optim = train_config['optim'],
            save_steps = train_config['save_steps'],
            logging_steps = train_config['logging_steps'],
            learning_rate = train_config['learning_rate'],
            weight_decay = train_config['weight_decay'],
            fp16 = train_config['fp16'],
            bf16 = train_config['bf16'],
            max_grad_norm = train_config['max_grad_norm'],
            max_steps = train_config['max_steps'],
            warmup_ratio = train_config['warmup_ratio'],
            group_by_length = train_config['group_by_length'],
            lr_scheduler_type = train_config['lr_scheduler_type'],
            report_to = train_config['report_to']
        )
    
    def tune_and_save(self, train_dataset, save_name: str):
        llama_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        llama_tokenizer.pad_token = llama_tokenizer.eos_token
        llama_tokenizer.padding_side = "right"

        self.tuner = SFTTrainer(
            model = self.base_model,
            train_dataset = train_dataset,
            peft_config = self.peft_parameters,
            dataset_text_field = "text",
            tokenizer = llama_tokenizer,
            args = self.train_params
        )

        self.tuner.train()
        self.tuner.model.save_pretrained(save_name)

        def eval_model(self, eval_prompts: list):
            '''
            Evaluate the model on a list of prompts
            '''
            pass
