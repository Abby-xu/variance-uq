from transformers import BertModel, BertTokenizer

def get_bert_model():
    return BertModel.from_pretrained("bert-base-uncased")

def get_bert_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")