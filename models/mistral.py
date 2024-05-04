from transformers import AutoModel, AutoTokenizer

def get_mistral_model():
    return AutoModel.from_pretrained('llmrails/ember-v1')
    # return AutoModel.from_pretrained("Salesforce/SFR-Embedding-Mistral")

def get_mistral_tokenizer():
    return AutoTokenizer.from_pretrained('llmrails/ember-v1')
    # return AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-Mistral")