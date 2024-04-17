from transformers import AutoModel, AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import config

device = config.device

def get_entailment_model():
    return AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").to(device)

def get_entailment_tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")