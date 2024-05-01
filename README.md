# LLM_Fine-Tuning-Optimization

This repo focuses on making the fine-tuning of LLMs esay. There is a wrapper that is developed on top of hugging face libraries that utilizes LoRA to fine tune the LLMs. This repo makes fine tuning so simple that it can be done by just selecting the model and the train files.

## Installation

### Cloning the Repository
To use the scripts, first clone the repository to your local machine:

```bash
!git clone https://github.com/nishantkushwaha-1999/llm_fine_tuning_optimization.git
```

### Installing Dependencies
After cloning, navigate to the directory and install the required Python packages:

```bash
pip install -r ./llm_fine_tuning_optimization/requirements.txt
```


## Usage
To start fine-tuning the models, ensure you have the necessary data and configurations set up, and run the Python scripts located in the repository. The main entry point is typically a script that orchestrates the fine-tuning process.

### Example Command
Refer below for specifying the tuning parameters and passing the data.

```python
from Tunner import fine_tunner
model_name = 'NousResearch/Llama-2-7b-chat-hf'
save_name = './drive/MyDrive/llama-2-7b-meddata-enhanced'

quant_config = {
    'load_in_4bit': True,
    'bnb_4bit_quant_type': "nf4",
    'bnb_4bit_compute_dtype': getattr(torch, "float16"),
    'bnb_4bit_use_double_quant': False
}

lora_config = {
    'lora_alpha': 16,
    'lora_dropout': 0.1,
    'r': 64,
    'bias': "none",
    'task_type': "CASUAL_LM",
    # use None if you dont wat to use target modules.
    'target_modules': [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ]
}

train_config = {
    'output_dir': "./results",
    'num_train_epochs': 1,
    'per_device_train_batch_size': 4,
    'gradient_accumulation_steps': 1,
    'optim': "paged_adamw_32bit",
    'save_steps': 25,
    'logging_steps': 25,
    'learning_rate': 2e-4,
    'weight_decay': 0.001,
    'fp16': False,
    'bf16': False,
    'max_grad_norm': 0.3,
    'max_steps': -1,
    'warmup_ratio': 0.03,
    'group_by_length': True,
    'lr_scheduler_type': "constant",
    'report_to': "tensorboard",
    'remove_unused_columns': False
}

tuner.load_model(model_name=model_name, quant_config=quant_config, lora_config=lora_config, train_config=train_config)
tuner.tune_and_save(train_data, save_name)
```
