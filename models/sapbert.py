from transformers import AutoModel, AutoTokenizer

def get_sapbert_model():
    return AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")

def get_sapbert_tokenizer():
    return AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")