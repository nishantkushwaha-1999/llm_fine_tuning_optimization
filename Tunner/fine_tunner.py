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

class Tlama_Tuner():
    def __init__(self):
        pass

    def to_dataset(self, instruction: list, response: list):
        assert len(instruction) == len(response)
        
        dataset = []
        for i in range(len(instruction)):
            item = {}
            item['text'] = "<s>[INST] " + str(instruction[i]) + " [/INST] " + str(response[i]) + " </s>"
            dataset.append(item)
        
        return iter(dataset)