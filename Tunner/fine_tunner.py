import torch
import itertools
import pyter
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu, sentence_bleu
from datasets import Dataset
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
            item = "<s>[INST] " + str(instruction[i]) + " [/INST] " + str(response[i]) + " </s>"
            dataset.append(item)
        
        dataset = {'text': dataset}
        dataset = Dataset.from_dict(dataset)
        return dataset
    
    def load_model(self, model_name: str, quant_config: dict, lora_config: dict, train_config: dict, ret: bool = False):
        self.model_name = model_name
        quant_config = BitsAndBytesConfig(
            load_in_4bit = quant_config['load_in_4bit'],
            bnb_4bit_quant_type = quant_config['bnb_4bit_quant_type'],
            bnb_4bit_compute_dtype = quant_config['bnb_4bit_compute_dtype'],
            bnb_4bit_use_double_quant = quant_config['bnb_4bit_use_double_quant']
            )
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map={"": 0}
        )

        self.base_model.config.use_cache = False
        self.base_model.config.pretraining_tp = 1

        self.peft_parameters = LoraConfig(
            lora_alpha = lora_config['lora_alpha'],
            lora_dropout = lora_config['lora_dropout'],
            r = lora_config['r'],
            bias = lora_config['bias'],
            task_type = lora_config['task_type'],
            target_modules = lora_config['target_modules']
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
            report_to = train_config['report_to'],
            remove_unused_columns = train_config['remove_unused_columns']
        )
    
    def tune_and_save(self, train_dataset, save_name: str):
        self.save_name = save_name

        self.llama_tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"

        self.tuner = SFTTrainer(
            model = self.base_model,
            train_dataset = train_dataset,
            peft_config = self.peft_parameters,
            dataset_text_field = "text",
            max_seq_length = None,
            tokenizer = self.llama_tokenizer,
            args = self.train_params,
            packing=False
        )

        self.tuner.train()
        self.tuner.model.save_pretrained(save_name)
        self.tuner.tokenizer.save_pretrained(save_name)

    def generate_base_text(self, query: str):
        text_gen = pipeline(task="text-generation", model=self.base_model, tokenizer=self.llama_tokenizer, max_length=200)
        output = text_gen(f"<s>[INST] {query} [/INST]")
        return output[0]['generated_text']

    def generate_text(self, query: str, use_trained_model: bool = True, model_name: str = None):
        if not use_trained_model:
            if model_name == None:
                raise ValueError("Please provide a model path")
            else:
                gen_model = model_name
        else:
            gen_model = self.tuner.model
        
        text_gen = pipeline(task="text-generation", model=gen_model, tokenizer=self.llama_tokenizer, max_length=200)
        output = text_gen(f"<s>[INST] {query} [/INST]")
        return output[0]['generated_text']
    
    def bleu(self, ref, gen):
        ref_bleu = []
        gen_bleu = []
        for l in gen:
            gen_bleu.append(l.split())
        for i,l in enumerate(ref):
            ref_bleu.append([l.split()])
        cc = SmoothingFunction()
        score_bleu = corpus_bleu(ref_bleu, gen_bleu, weights=(0, 1, 0, 0), smoothing_function=cc.method4)
        return score_bleu
    
    def ter(self, ref, gen):
        if len(ref) == 1:
            total_score =  pyter.ter(gen[0].split(), ref[0].split())
        else:
            total_score = 0
            for i in range(len(gen)):
                total_score = total_score + pyter.ter(gen[i].split(), ref[i].split())
            total_score = total_score/len(gen)
        return total_score
    
    def __split_into_words(self, sentences):
        return list(itertools.chain(*[_.split(" ") for _ in sentences]))

    def __get_word_ngrams(self, n, sentences):
        """Calculates word n-grams for multiple sentences.
        """
        assert len(sentences) > 0
        assert n > 0

        words = self.__split_into_words(sentences)
        return self.__get_ngrams(n, words)

    def __get_ngrams(self, n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i:i + n]))
        return ngram_set

    def rouge_n(self, reference_sentences, evaluated_sentences, n=2):
        if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
            raise ValueError("Collections must contain at least 1 sentence.")

        evaluated_ngrams = self.__get_word_ngrams(n, evaluated_sentences)
        reference_ngrams = self.__get_word_ngrams(n, reference_sentences)
        reference_count = len(reference_ngrams)
        evaluated_count = len(evaluated_ngrams)

        overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
        overlapping_count = len(overlapping_ngrams)

        if evaluated_count == 0:
            precision = 0.0
        else:
            precision = overlapping_count / evaluated_count

        if reference_count == 0:
            recall = 0.0
        else:
            recall = overlapping_count / reference_count

        f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

        return recall

    def eval_model(self, ref_text: list, gen_text: list, metrics: list):
        assert len(ref_text) == len(gen_text)

        ref = []
        gen = []
        for i in range(len(ref_text)):
            ref.append(ref_text[i].strip())
            gen.append(gen_text[i].strip())
        
        scores = []
        for metric in metrics:
            if metric == 'bleu':
                scores.append(self.bleu(ref, gen))
            elif metric == 'rouge':
                scores.append(self.rouge_n(ref, gen))
            elif metric == 'ter':
                scores.append(self.ter(ref, gen))
            else:
                raise ValueError("Invalid metric")
        
        return tuple(scores)