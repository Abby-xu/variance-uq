from transformers import AutoModel, AutoTokenizer

def get_entailment_model():
    return AutoModel.from_pretrained("microsoft/deberta-v2-xlarge-mnli")

def get_entailment_tokenizer():
    return AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")