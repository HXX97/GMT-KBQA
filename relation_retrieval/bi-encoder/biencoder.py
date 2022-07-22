import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
from transformers import AutoModel


class BiEncoderModule(torch.nn.Module):
    def __init__(self, device, bert_model="bert-base-uncased", tokenizer=None, freeze_bert=False):
        super(BiEncoderModule, self).__init__()
        self.question_bert_layer = AutoModel.from_pretrained(bert_model)
        self.relation_bert_layer = AutoModel.from_pretrained(bert_model)
        self.device = device
        if tokenizer:
            self.question_bert_layer.resize_token_embeddings(len(tokenizer))
            self.relation_bert_layer.resize_token_embeddings(len(tokenizer))
        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.question_bert_layer.parameters():
                p.requires_grad = False
            for p in self.relation_bert_layer.parameters():
                p.requires_grad = False     

    @autocast()
    def forward(
        self,
        question_input_ids,
        question_attn_masks,
        question_token_type_ids,
        relations_input_ids,
        relations_attn_masks,
        relations_token_type_ids,
        golden_id
    ):
        embedding_question = self.question_bert_layer(question_input_ids, question_attn_masks, question_token_type_ids).pooler_output
            
        
        embedding_relations = []
        # bert only accept (batch_size, maxlen) size of input, while embedding_relations is with the size (batch_size, sample_size, maxlen)
        for i in range(0, relations_input_ids.shape[1]):
            relation_input_id = relations_input_ids[:,i,:]
            # print('relation_input_id: {}'.format(relation_input_id.shape)) # (batch_size, maxlen)
            relations_attn_mask = relations_attn_masks[:,i,:]
            relations_token_type_id = relations_token_type_ids[:,i,:]
            embedding_relation = self.relation_bert_layer(relation_input_id, relations_attn_mask, relations_token_type_id).pooler_output
            embedding_relations.append(embedding_relation)
        
        embedding_relations = torch.stack(embedding_relations, dim=1) # 已确认
        
        embedding_question = embedding_question.unsqueeze(1)
        # print('embedding_question: {}'.format(embedding_question.shape)) # (batch_size, 1, 768)
        # print('embedding_relations: {}'.format(embedding_relations.shape)) # (batch_size, sample_size, 768)
        
        scores = torch.bmm(embedding_question, torch.transpose(embedding_relations, 1, 2)).squeeze(1)
        # print('scores: {}'.format(scores.shape)) # (batch_size, sample_size)
        loss = self.calculate_loss(scores, golden_id)
        
        return scores.to(self.device), loss
    
    @autocast()
    def calculate_loss(self, scores, golden_id):
        """
        scores: (batch_size, sample_size)
        golden_id: (batch_size)
        loss = -scores[golden_id] + log \sum_{i=1}^B exp(scores[i])
        """
        assert len(golden_id.shape) == 1, print(golden_id.shape)
        assert golden_id.shape[0] == scores.shape[0], print('golden_id: {}, scores: {}'.format(golden_id.shape, scores.shape))
        
        loss_fct = nn.CrossEntropyLoss()
        # target = torch.zeros(scores.shape)
        # for i in range(0, golden_id.shape[0]):
        #     target[i][golden_id[i]] = 1 # one-hot encoding target, golden_id --> 1, others --> 0
        # print('golden id: {}'.format(golden_id))
        # loss = loss_fct(scores.to(self.device), target.to(self.device)) / scores.shape[0]
        # loss = loss_fct(scores.to(self.device), target.long().to(self.device)) / scores.shape[0]
        loss = loss_fct(scores.to(self.device), golden_id.to(self.device)) / scores.shape[0]
        
        return loss

    def encode_question(self, question_token_ids, question_attn_masks, question_token_type_ids):
        """
        question_token_ids: (batch_size, maxlen)
        """
        question_representation = self.question_bert_layer(question_token_ids, question_attn_masks, question_token_type_ids).pooler_output # (batch_size, 768)
        return question_representation

    def encode_relation(self, relation_input_id, relations_attn_mask, relations_token_type_id):
        """
        relation_input_id: (batch_size, maxlen)
        """
        relation_representation = self.relation_bert_layer(relation_input_id, relations_attn_mask, relations_token_type_id).pooler_output # (batch_size, 768)
        return relation_representation