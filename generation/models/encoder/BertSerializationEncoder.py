import torch
import torch.nn as nn
from transformers import BertModel

class BertSerializationEncoder(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', freeze_bert=True, verbose=False):
        super(BertSerializationEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.verbose=verbose
        for param in self.bert.parameters():
            param.requires_grad = not freeze_bert

    def forward(self, x):
        embedding = self.bert(x)
        if self.verbose:
            print("x: {}".format(x.shape)) # (batch_size, fix_length)
            print("embedding: {}".format(embedding)) # no shape, has attribute like `hidden_states`,`attentions` and so on
            print("embedding[0]: {}".format(embedding[0].shape)) # (batch_size, fix_length, bert_embedding_dim(768))
        return embedding