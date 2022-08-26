import numpy as np
import torch
import torch.nn as nn
from transformers import T5Config, T5ForConditionalGeneration, AutoTokenizer
import sys
sys.path.append('..')
# from structure_filling_data_process import filter_syntax_error, post_process


def list_reshape(prev_list, dim0, dim1):
    assert len(prev_list) == dim0 * dim1
    new_list = [prev_list[i*dim1: (i+1)*dim1] for i in range(dim0)]
    return new_list


def filter_candidates_by_classification_results(candidates, classification_results, cross_entropy_loss=False, extract_label=False, cls_labels=None):
    """
    candidates: list of [dim0, dim1]
    classification_results: tensor of [dim0, dim1]
    filter condition: logits > 0 if BCELoss, logits = 1.0 if CrossEntropyLoss
    cls_labels: if given, get intersection between classification result and golden label
    """
    assert len(candidates) == classification_results.shape[0]
    assert len(candidates[0]) == classification_results.shape[1], print(len(candidates[0]), classification_results.shape[1])

    if cross_entropy_loss:
        indices = np.argwhere(classification_results.detach().cpu().numpy() == 1.0).tolist()
    else:
        indices = np.argwhere(classification_results.detach().cpu().numpy() > 0.0).tolist()
    
    if cls_labels is not None:
        golden_indices = np.argwhere(cls_labels.detach().cpu().numpy() == 1.0).tolist()


    filtered_candidates = []
    for i in range(len(candidates)):
        row = []
        for j in range(len(candidates[0])):
            if cls_labels is not None:
                if [i,j] in indices and [i,j] in golden_indices:
                    if extract_label:
                        row.append(candidates[i][j]['label'])
                    else:
                        row.append(candidates[i][j])
            else:
                if [i,j] in indices:
                    if extract_label:
                        row.append(candidates[i][j]['label'])
                    else:
                        row.append(candidates[i][j])
        filtered_candidates.append(row)
    
    return filtered_candidates


def _textualize_relation(r):
    """return a relation string with '_' and '.' replaced"""
    if "_" in r: # replace "_" with " "
        r = r.replace("_", " ")
    if "." in r: # replace "." with " , "
        r = r.replace(".", " , ")
    return r


class T5_generation(nn.Module):
    def __init__(
        self,
        pretrained_model_path,
        is_test=False,
        max_tgt_len=196,
    ):
        super().__init__()
        self._is_test = is_test
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
        self.max_tgt_len = max_tgt_len
    
    def forward(
        self,
        input_ids_gen=None,
        gen_labels=None,
        gen_attention_mask=None,
    ):
        gen_outputs = self.t5(
            input_ids=input_ids_gen,
            attention_mask=gen_attention_mask,
            labels=gen_labels,
        )
        gen_loss = gen_outputs['loss']

        return gen_loss
    
    def inference(
        self,
        input_ids_gen=None,
        gen_attention_mask=None,
        num_beams=5,
    ):
        with torch.no_grad():
            gen_outputs = self.t5.generate(
                input_ids=input_ids_gen,
                attention_mask=gen_attention_mask,
                use_cache=True,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=self.max_tgt_len
            )
            gen_outputs = torch.reshape(gen_outputs,(input_ids_gen.size(0),num_beams,-1))
        
        return gen_outputs


class T5_generation_concat(nn.Module):
    """
    only differs in input format compared to T5_generation
    Concat
    - relation with logit > 0 from Cross-Encoder output
    - candidate entities after disambiguatio
    - or golden relation/entity
    """
    def __init__(
        self,
        pretrained_model_path,
        is_test=False,
        max_tgt_len=196,
    ):
        super().__init__()
        self._is_test = is_test
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
        self.max_tgt_len = max_tgt_len
    
    def forward(
        self,
        input_ids_gen=None,
        gen_labels=None,
        gen_attention_mask=None,
    ):
        gen_outputs = self.t5(
            input_ids=input_ids_gen,
            attention_mask=gen_attention_mask,
            labels=gen_labels,
        )
        gen_loss = gen_outputs['loss']

        return gen_loss
    
    def inference(
        self,
        input_ids_gen=None,
        gen_attention_mask=None,
        num_beams=5,
    ):
        with torch.no_grad():
            gen_outputs = self.t5.generate(
                input_ids=input_ids_gen,
                attention_mask=gen_attention_mask,
                use_cache=True,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=self.max_tgt_len
            )
            gen_outputs = torch.reshape(gen_outputs,(input_ids_gen.size(0),num_beams,-1))
        
        return gen_outputs


class T5_Multitask_Relation_Concat(nn.Module):
    """ 
    Relation classification + Generation
    Input of Generation task: q [REL] r1 [REL] r2
    """
    def __init__(
        self, 
        pretrained_model_path, 
        device='cuda', 
        max_src_len=128, 
        max_tgt_len=196, 
        tokenizer=None, 
        is_test=False, 
        sample_size=10, 
        do_lower=False, 
        cross_entropy_loss=False, 
        add_prefix=False,
    ):
        super().__init__()
        self._is_test = is_test
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
        self.device = device
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(pretrained_model_path)
        self.sample_size = sample_size # number of candidate relations
        self.do_lower = do_lower
        self.add_prefix = add_prefix

        if 't5-large' in pretrained_model_path.lower():
            hidden_size = 1024
        elif 't5-small' in pretrained_model_path.lower():
            hidden_size = 512
        else:
            hidden_size = 768

        self.dropout = nn.Dropout(0.1)

        self.cross_entropy_loss = cross_entropy_loss
        if self.cross_entropy_loss:
            self.cls_layer = nn.Linear(hidden_size, 2)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.cls_layer = nn.Linear(hidden_size, 1) # binary classification
            self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self,
        input_ids_gen=None,
        input_ids_clf=None,
        gen_labels=None,
        clf_labels=None,
        clf_attention_mask=None,
        textual_candidate_relations=None,
        textual_input_src_gen=None,
        normalize_relations=False,
        do_concat=False
    ):
        # use T5 encoder to encode input_ids
        # clf_encoder_outputs: [batch_size, max_len, hidden_size]
        textual_candidate_relations = list_reshape(textual_candidate_relations, int(len(textual_candidate_relations)/self.sample_size), self.sample_size)
        clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_clf, attention_mask=clf_attention_mask)
        if clf_labels is not None:
            # sentence_embedding: [batch_size*sample_size, embedding_dim]
            sentence_embedding = torch.mean(clf_encoder_outputs.last_hidden_state,dim=1)
            # [batch_size*sample_size, 1] if BCELoss, [batch_size*sample_size, 2] if CrossEntropy
            clf_predict_logits = self.cls_layer(self.dropout(sentence_embedding))
            
            # classification loss
            if self.cross_entropy_loss:
                clf_loss = self.criterion(clf_predict_logits.float(), clf_labels)
            else:
                clf_loss = self.criterion(clf_predict_logits.float(), clf_labels.unsqueeze(1).float())
            
            clf_predict_logits = torch.reshape(clf_predict_logits, (input_ids_gen.size(0), self.sample_size, -1))
            if self.cross_entropy_loss:
                clf_predict_logits = torch.argmax(clf_predict_logits, dim=2)
            else:
                clf_predict_logits = clf_predict_logits.squeeze(2)

            
            filtered_candidate_relations = filter_candidates_by_classification_results(
                textual_candidate_relations,
                clf_predict_logits,
                cross_entropy_loss=self.cross_entropy_loss
            )
        else:
            clf_loss = None
            filtered_candidate_relations = None
        
        input_ids_gen_concatenated = []
        for (input_src, cand_rels) in zip(textual_input_src_gen, filtered_candidate_relations):
            if self.add_prefix:
                input_src = 'Translate to S-Expression: ' + input_src
            if do_concat:
                for rel in cand_rels:
                    if normalize_relations:
                        input_src += " [REL] " + _textualize_relation(rel)
                    else:
                        input_src += " [REL] " + rel
            if self.do_lower:
                input_src = input_src.lower()

            tokenized_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_gen_concatenated.append(tokenized_src)
        
        # dynamic mini-batch padding
        src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')
        
        gen_outputs = self.t5(
            input_ids=src_encoded['input_ids'].to(self.device),
            attention_mask=src_encoded['attention_mask'].to(self.device),
            labels=gen_labels,
        )

        gen_loss = gen_outputs['loss']

        if clf_loss is not None:
            total_loss = gen_loss + clf_loss
        else:
            total_loss = gen_loss

        return total_loss
    
    # TODO:
    def inference(self,
        input_ids_gen=None,
        input_ids_clf=None,
        clf_attention_mask=None,
        num_beams=5,
        clf_sample_size=10,
        textual_candidate_relations=None,
        textual_input_src_gen=None,
        normalize_relations=False,
    ):
        with torch.no_grad():
            textual_candidate_relations = list_reshape(textual_candidate_relations, int(len(textual_candidate_relations)/self.sample_size), self.sample_size)
            
            clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_clf, attention_mask=clf_attention_mask)
            sentence_embedding = torch.mean(clf_encoder_outputs.last_hidden_state,dim=1)
            # [batch_size*sample_size, 1] if BCELoss, [batch_size*sample_size, 2] if CrossEntropy
            clf_predict_logits = self.cls_layer(self.dropout(sentence_embedding))
            # clf_outputs: [batch_size, sample_size, 1/2]
            clf_outputs = torch.reshape(clf_predict_logits,(input_ids_gen.size(0),clf_sample_size,-1))
            # logits, reshape to [batch_size, sample_size]
            if self.cross_entropy_loss:
                clf_predict_logits = torch.argmax(clf_outputs, dim=2)
                clf_outputs = torch.argmax(clf_outputs, dim=2)
            else:
                clf_predict_logits = clf_outputs.squeeze(2)
                clf_outputs = clf_outputs.squeeze(2)
            
            filtered_candidate_relations = filter_candidates_by_classification_results(
                textual_candidate_relations,
                clf_predict_logits,
                cross_entropy_loss=self.cross_entropy_loss
            )

            input_ids_gen_concatenated = []
            for (input_src, cand_rels) in zip(textual_input_src_gen, filtered_candidate_relations):
                if self.add_prefix:
                    input_src = 'Translate to S-Expression: ' + input_src
                for rel in cand_rels:
                    if normalize_relations:
                        input_src += " [REL] " + _textualize_relation(rel)
                    else:
                        input_src += " [REL] " + rel

                if self.do_lower:
                    input_src = input_src.lower()

                tokenized_src = self.tokenizer(
                    input_src,
                    max_length=self.max_src_len,
                    truncation=True,
                    return_tensors='pt',
                ).data['input_ids'].squeeze(0)
                input_ids_gen_concatenated.append(tokenized_src)
            
            # dynamic mini-batch padding
            src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')
            
            gen_outputs = self.t5.generate(
                input_ids=src_encoded['input_ids'].to(self.device),
                attention_mask=src_encoded['attention_mask'].to(self.device),
                use_cache=True,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=self.max_tgt_len
            )
            # src_encoded['input_ids']: [batch_size, padding_length]
            # [batch_size, num_beams, -1]
            gen_outputs = torch.reshape(gen_outputs,(src_encoded['input_ids'].shape[0],num_beams,-1))
            # gen_outputs = [p.cpu().numpy() for p in gen_outputs]

        return gen_outputs,clf_outputs



class T5_Multitask_Entity_Concat(nn.Module):
    """
    Entity disambiguation + Generation
    """
    def __init__(
        self, 
        pretrained_model_path, 
        device='cuda', 
        max_src_len=128, 
        max_tgt_len=196, 
        tokenizer=None, 
        is_test=False, 
        sample_size=10, 
        do_lower=False, 
        add_prefix=False,
    ):
        super().__init__()
        self._is_test = is_test
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
        self.device = device
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(pretrained_model_path)
        self.sample_size = sample_size # number of candidate relations
        self.do_lower = do_lower
        self.add_prefix = add_prefix

        if 't5-large' in pretrained_model_path.lower():
            hidden_size = 1024
        elif 't5-small' in pretrained_model_path.lower():
            hidden_size = 512
        else:
            hidden_size = 768

        self.dropout = nn.Dropout(0.1)
        self.cls_layer = nn.Linear(hidden_size, 1) # binary classification
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self,
        input_ids_gen=None,
        input_ids_clf=None,
        gen_labels=None,
        clf_labels=None,
        clf_attention_mask=None,
        textual_candidate_entities=None,
        textual_input_src_gen=None,
        do_concat=False
    ):
        textual_candidate_entities = list_reshape(
            textual_candidate_entities,
            int(len(textual_candidate_entities) / self.sample_size),
            self.sample_size
        )
        clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_clf, attention_mask=clf_attention_mask)
        if clf_labels is not None:
            # sentence_embedding: [batch_size*sample_size, embedding_dim]
            sentence_embedding = torch.mean(clf_encoder_outputs.last_hidden_state,dim=1)
            # [batch_size*sample_size, 1] if BCELoss, [batch_size*sample_size, 2] if CrossEntropy
            clf_predict_logits = self.cls_layer(self.dropout(sentence_embedding))
            # classification loss
            clf_loss = self.criterion(clf_predict_logits.float(), clf_labels.unsqueeze(1).float())
            clf_predict_logits = torch.reshape(clf_predict_logits, (input_ids_gen.size(0), self.sample_size))

            filtered_candidate_entities = filter_candidates_by_classification_results(
                textual_candidate_entities,
                clf_predict_logits,
                extract_label=True
            )
        else:
            clf_loss = None
            filtered_candidate_entities = None
        
        input_ids_gen_concatenated = []
        for (input_src, cand_ents) in zip(textual_input_src_gen, filtered_candidate_entities):
            if self.add_prefix:
                input_src = 'Translate to S-Expression: ' + input_src
            if do_concat:
                for ent in cand_ents:
                    input_src += " [ENT] " + ent
            if self.do_lower:
                input_src = input_src.lower()

            tokenized_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_gen_concatenated.append(tokenized_src)

        # dynamic mini-batch padding
        src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')
        # TODO, maybe we can change the input according to the classification results
        gen_outputs = self.t5(
            input_ids=src_encoded['input_ids'].to(self.device),
            attention_mask=src_encoded['attention_mask'].to(self.device),
            labels=gen_labels,
        )

        gen_loss = gen_outputs['loss']

        if clf_loss is not None:
            total_loss = gen_loss + clf_loss
        else:
            total_loss = gen_loss

        return total_loss
    
    def inference(
        self,
        input_ids_gen=None,
        input_ids_clf=None,
        clf_attention_mask=None,
        num_beams=5,
        textual_candidate_entities=None,
        textual_input_src_gen=None,
    ):
        with torch.no_grad():
            textual_candidate_entities = list_reshape(
                textual_candidate_entities,
                int(len(textual_candidate_entities) / self.sample_size),
                self.sample_size
            )
            clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_clf, attention_mask=clf_attention_mask)
            sentence_embedding = torch.mean(clf_encoder_outputs.last_hidden_state,dim=1)
            clf_predict_logits = self.cls_layer(self.dropout(sentence_embedding))
            clf_outputs = torch.reshape(clf_predict_logits,(input_ids_gen.size(0), self.sample_size))
            filtered_candidate_entities = filter_candidates_by_classification_results(
                textual_candidate_entities,
                clf_outputs,
                extract_label=True
            )

            input_ids_gen_concatenated = []
            for (input_src, cane_ents) in zip(textual_input_src_gen, filtered_candidate_entities):
                if self.add_prefix:
                    input_src = 'Translate to S-Expression: ' + input_src
                for ent in cane_ents:
                    input_src += " [ENT] " + ent
                if self.do_lower:
                    input_src = input_src.lower()
                tokenized_src = self.tokenizer(
                    input_src,
                    max_length=self.max_src_len,
                    truncation=True,
                    return_tensors='pt',
                ).data['input_ids'].squeeze(0)
                input_ids_gen_concatenated.append(tokenized_src)
            
            src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')

            gen_outputs = self.t5.generate(
                input_ids=src_encoded['input_ids'].to(self.device),
                attention_mask=src_encoded['attention_mask'].to(self.device),
                use_cache=True,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=self.max_tgt_len
            )
            gen_outputs = torch.reshape(gen_outputs,(src_encoded['input_ids'].shape[0],num_beams,-1))

        return gen_outputs,clf_outputs

class T5_MultiTask_Relation_Entity_Concat(nn.Module):
    """ 
    Relation classification + Entity disambiguation + Generation
    """
    def __init__(
        self, 
        pretrained_model_path, 
        device='cuda', 
        max_src_len=128,
        max_tgt_len=196, 
        tokenizer=None, 
        is_test=False, 
        sample_size=10, 
        do_lower=False, 
        cross_entropy_loss=False,
        add_prefix=False,
        train_concat_true=False
    ):
        super().__init__()
        self._is_test = is_test
        self.t5 = T5ForConditionalGeneration.from_pretrained(pretrained_model_path)
        self.device = device
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(pretrained_model_path)
        self.REL_TOKEN = ' [REL] '
        self.ENT_TOKEN = ' [ENT] '
        self.sample_size = sample_size # number of candidate relations and candidate entities
        self.do_lower = do_lower
        self.add_prefix = add_prefix
        # during training, only concat true classified results(compared with golden labels); during testing, concat normally(anything prev tasks output)
        # Been proved a bad practice
        self.train_concat_true = train_concat_true

        if 't5-large' in pretrained_model_path.lower():
            hidden_size = 1024
        elif 't5-small' in pretrained_model_path.lower():
            hidden_size = 512
        else:
            hidden_size = 768

        self.dropout = nn.Dropout(0.1)

        self.cross_entropy_loss = cross_entropy_loss
        if self.cross_entropy_loss:
            self.cls_layer = nn.Linear(hidden_size, 2)
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.cls_layer = nn.Linear(hidden_size, 1) # binary classification
            self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self,
        input_ids_gen=None,
        input_ids_relation_clf=None,
        gen_labels=None,
        relation_clf_labels=None,
        entity_clf_labels=None,
        relation_clf_attention_mask=None,
        textual_candidate_relations=None,
        textual_input_src_gen=None,
        normalize_relations=False,
        rich_textual_candidate_entities_list=None,
        do_concat=False
    ):
        # clf_encoder_outputs: [batch_size, max_len, hidden_size]
        textual_candidate_relations = list_reshape(textual_candidate_relations, int(len(textual_candidate_relations)/self.sample_size), self.sample_size)
        rich_textual_candidate_entities_list = list_reshape(rich_textual_candidate_entities_list, int(len(rich_textual_candidate_entities_list) / self.sample_size), self.sample_size) # list of [batch_size, sample_size]

        # Task 1: Relation classification
        relation_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_relation_clf, attention_mask=relation_clf_attention_mask)
        if relation_clf_labels is not None:
            # sentence_embedding: [batch_size*sample_size, embedding_dim]
            relation_sentence_embedding = torch.mean(relation_clf_encoder_outputs.last_hidden_state,dim=1)

            # [batch_size*sample_size, 1] if BCELoss, [batch_size*sample_size, 2] if CrossEntropy
            relation_clf_predict_logits = self.cls_layer(self.dropout(relation_sentence_embedding))
            
            # classification loss
            if self.cross_entropy_loss:
                relation_clf_loss = self.criterion(relation_clf_predict_logits.float(), relation_clf_labels)
            else:
                relation_clf_loss = self.criterion(relation_clf_predict_logits.float(), relation_clf_labels.unsqueeze(1).float())
            
            relation_clf_predict_logits = torch.reshape(relation_clf_predict_logits, (input_ids_gen.size(0), self.sample_size, -1))
            if self.cross_entropy_loss:
                relation_clf_predict_logits = torch.argmax(relation_clf_predict_logits, dim=2)
            else:
                relation_clf_predict_logits = relation_clf_predict_logits.squeeze(2)
            
            if self.train_concat_true:
                filtered_candidate_relations = filter_candidates_by_classification_results(
                    textual_candidate_relations,
                    relation_clf_predict_logits,
                    cross_entropy_loss=self.cross_entropy_loss,
                    cls_labels=torch.reshape(relation_clf_labels, (input_ids_gen.size(0), self.sample_size))
                )
            else:
                print('relation train_concat_true: False')
                filtered_candidate_relations = filter_candidates_by_classification_results(
                    textual_candidate_relations,
                    relation_clf_predict_logits,
                    cross_entropy_loss=self.cross_entropy_loss
                )
        else:
            relation_clf_loss = None
            filtered_candidate_relations = None     

        # Task2: Entity classification
        # for each entity e, concat intersection of (e's one-hop relations, filtered_candidate_relations)
        input_ids_entity_concatenated = []
        for (input_src, cand_ents, cand_relations) in zip(textual_input_src_gen, rich_textual_candidate_entities_list, filtered_candidate_relations):
            ent_src = input_src
            if self.add_prefix:
                ent_src = 'Entity Classification: ' + ent_src
            if self.do_lower:
                ent_src = ent_src.lower()

            for cand_ent in cand_ents:
                ent_info = cand_ent['label']
                intersec_relations = list(set(cand_ent["1hop_relations"]) & set(cand_relations))
                for rel in intersec_relations:
                    if normalize_relations:
                        ent_info += ("|" + _textualize_relation(rel))
                    else:
                        ent_info += ("|" + rel)
                if self.do_lower:
                    ent_info = ent_info.lower()
                tokenized_entity_src = self.tokenizer(
                    ent_src,
                    ent_info,
                    max_length=self.max_src_len,
                    truncation='longest_first',
                    return_tensors='pt',
                ).data['input_ids'].squeeze(0)
                input_ids_entity_concatenated.append(tokenized_entity_src)
    
        # dynamic mini-batch padding
        entity_clf_encoded = self.tokenizer.pad({'input_ids': input_ids_entity_concatenated},return_tensors='pt')
        input_ids_entity_clf = entity_clf_encoded['input_ids'].to(self.device) # [batch_size*sample_size, max_len]
        entity_clf_attention_mask = entity_clf_encoded['attention_mask'].to(self.device) # [batch_size*sample_size, max_len]

        entity_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_entity_clf, attention_mask=entity_clf_attention_mask)
        if entity_clf_labels is not None:
            # sentence_embedding: [batch_size*sample_size, embedding_dim]
            entity_sentence_embedding = torch.mean(entity_clf_encoder_outputs.last_hidden_state,dim=1)

            # [batch_size*sample_size, 1] if BCELoss, [batch_size*sample_size, 2] if CrossEntropy
            entity_clf_predict_logits = self.cls_layer(self.dropout(entity_sentence_embedding))
            
            # classification loss
            if self.cross_entropy_loss:
                entity_clf_loss = self.criterion(entity_clf_predict_logits.float(), entity_clf_labels)
            else:
                entity_clf_loss = self.criterion(entity_clf_predict_logits.float(), entity_clf_labels.unsqueeze(1).float())
            
            entity_clf_predict_logits = torch.reshape(entity_clf_predict_logits, (input_ids_gen.size(0), self.sample_size, -1))
            if self.cross_entropy_loss:
                entity_clf_predict_logits = torch.argmax(entity_clf_predict_logits, dim=2)
            else:
                entity_clf_predict_logits = entity_clf_predict_logits.squeeze(2)
            
            if self.train_concat_true:
                filtered_candidate_entities = filter_candidates_by_classification_results(
                    rich_textual_candidate_entities_list,
                    entity_clf_predict_logits,
                    cross_entropy_loss=self.cross_entropy_loss,
                    extract_label=True,
                    cls_labels=torch.reshape(entity_clf_labels, (input_ids_gen.size(0), self.sample_size))
                )
            else:
                print('entity train_concat_true: False')
                filtered_candidate_entities = filter_candidates_by_classification_results(
                    rich_textual_candidate_entities_list,
                    entity_clf_predict_logits,
                    cross_entropy_loss=self.cross_entropy_loss,
                    extract_label=True
                )
        else:
            entity_clf_loss = None
            filtered_candidate_entities = None 

        input_ids_gen_concatenated = []
        for (input_src, cand_ents, cand_rels) in zip(textual_input_src_gen, filtered_candidate_entities, filtered_candidate_relations):
            if self.add_prefix:
                input_src = 'Translate to S-Expression: ' + input_src
            if do_concat:
                for rel in cand_rels:
                    if normalize_relations:
                        input_src += self.REL_TOKEN + _textualize_relation(rel)
                    else:
                        input_src += self.REL_TOKEN + rel
                for ent in cand_ents:
                    input_src += self.ENT_TOKEN + ent
            if self.do_lower:
                input_src = input_src.lower()

            tokenized_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_gen_concatenated.append(tokenized_src)
        
        # dynamic mini-batch padding
        src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')
        
        gen_outputs = self.t5(
            input_ids=src_encoded['input_ids'].to(self.device),
            attention_mask=src_encoded['attention_mask'].to(self.device),
            labels=gen_labels,
        )

        gen_loss = gen_outputs['loss']
        if entity_clf_loss is not None and relation_clf_loss is not None:
            total_loss = gen_loss + entity_clf_loss + relation_clf_loss
        else:
            total_loss = gen_loss

        return total_loss
    
    def inference(self,
        input_ids_gen=None,
        input_ids_relation_clf=None,
        relation_clf_attention_mask=None,
        num_beams=5,
        textual_candidate_relations=None,
        textual_input_src_gen=None,
        normalize_relations=False,
        rich_textual_candidate_entities_list=None
        ):
        textual_candidate_relations = list_reshape(textual_candidate_relations, int(len(textual_candidate_relations)/self.sample_size), self.sample_size)
        rich_textual_candidate_entities_list = list_reshape(rich_textual_candidate_entities_list, int(len(rich_textual_candidate_entities_list) / self.sample_size), self.sample_size) # list of [batch_size, sample_size]

        # 1. Get prediction from Relation Classification
        relation_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_relation_clf, attention_mask=relation_clf_attention_mask)
        relation_sentence_embedding = torch.mean(relation_clf_encoder_outputs.last_hidden_state,dim=1)
        # [batch_size*sample_size, 1] if BCELoss, [batch_size*sample_size, 2] if CrossEntropy
        relation_clf_predict_logits = self.cls_layer(self.dropout(relation_sentence_embedding))
        relation_clf_outputs = torch.reshape(relation_clf_predict_logits, (input_ids_gen.size(0), self.sample_size, -1)) # [batch_size, sample_size, 1/2]

        if self.cross_entropy_loss:
            relation_clf_predict_logits = torch.argmax(relation_clf_outputs, dim=2)
            relation_clf_outputs = torch.argmax(relation_clf_outputs, dim=2)
        else:
            relation_clf_predict_logits = relation_clf_outputs.squeeze(2)
            relation_clf_outputs = relation_clf_outputs.squeeze(2)
        
        filtered_candidate_relations = filter_candidates_by_classification_results(
            textual_candidate_relations,
            relation_clf_predict_logits,
            cross_entropy_loss=self.cross_entropy_loss
        )
       
        # 2. Get prediction from entity classification
        # for each entity e, concat intersection of (e's one-hop relations, filtered_candidate_relations)
        input_ids_entity_concatenated = []
        for (input_src, cand_ents, cand_relations) in zip(textual_input_src_gen, rich_textual_candidate_entities_list, filtered_candidate_relations):
            if self.add_prefix:
                input_src = 'Entity Classification: ' + input_src
            if self.do_lower:
                input_src = input_src.lower()
            for cand_ent in cand_ents:
                ent_info = cand_ent['label']
                intersec_relations = list(set(cand_ent["1hop_relations"]) & set(cand_relations))
                for rel in intersec_relations:
                    if normalize_relations:
                        ent_info += ("|" + _textualize_relation(rel))
                    else:
                        ent_info += ("|" + rel)
                if self.do_lower:
                    ent_info = ent_info.lower()
                tokenized_entity_src = self.tokenizer(
                    input_src,
                    ent_info,
                    max_length=self.max_src_len,
                    truncation='longest_first',
                    return_tensors='pt',
                ).data['input_ids'].squeeze(0)
                input_ids_entity_concatenated.append(tokenized_entity_src)
        # dynamic mini-batch padding
        entity_clf_encoded = self.tokenizer.pad({'input_ids': input_ids_entity_concatenated},return_tensors='pt')
        input_ids_entity_clf = entity_clf_encoded['input_ids'].to(self.device) # [batch_size*sample_size, max_len]
        entity_clf_attention_mask = entity_clf_encoded['attention_mask'].to(self.device) # [batch_size*sample_size, max_len]

        entity_clf_encoder_outputs = self.t5.encoder(input_ids=input_ids_entity_clf, attention_mask=entity_clf_attention_mask)
        entity_sentence_embedding = torch.mean(entity_clf_encoder_outputs.last_hidden_state,dim=1)
        # [batch_size*sample_size, 1] if BCELoss, [batch_size*sample_size, 2] if CrossEntropy
        entity_clf_predict_logits = self.cls_layer(self.dropout(entity_sentence_embedding))
        entity_clf_outputs = torch.reshape(entity_clf_predict_logits,(input_ids_gen.size(0), self.sample_size,-1)) # [batch_size, sample_size, 1]

        if self.cross_entropy_loss:
            entity_clf_predict_logits = torch.argmax(entity_clf_outputs, dim=2)
            entity_clf_outputs = torch.argmax(entity_clf_outputs, dim=2)
        else:
            entity_clf_predict_logits = entity_clf_outputs.squeeze(2)
            entity_clf_outputs = entity_clf_outputs.squeeze(2)
        

        filtered_candidate_entities = filter_candidates_by_classification_results(
            rich_textual_candidate_entities_list,
            entity_clf_predict_logits,
            cross_entropy_loss=self.cross_entropy_loss,
            extract_label=True
        )

        input_ids_gen_concatenated = []
        for (input_src, cand_ents, cand_rels) in zip(textual_input_src_gen, filtered_candidate_entities, filtered_candidate_relations):
            if self.add_prefix:
                input_src = 'Translate to S-Expression: ' + input_src
            for rel in cand_rels:
                if normalize_relations:
                    input_src += self.REL_TOKEN + _textualize_relation(rel)
                else:
                    input_src += self.REL_TOKEN + rel
            for ent in cand_ents:
                input_src += self.ENT_TOKEN + ent
            if self.do_lower:
                input_src = input_src.lower()

            tokenized_src = self.tokenizer(
                input_src,
                max_length=self.max_src_len,
                truncation=True,
                return_tensors='pt',
            ).data['input_ids'].squeeze(0)
            input_ids_gen_concatenated.append(tokenized_src)
        
        # dynamic mini-batch padding
        src_encoded = self.tokenizer.pad({'input_ids': input_ids_gen_concatenated},return_tensors='pt')
        
        with torch.no_grad():
            gen_outputs = self.t5.generate(
                input_ids=src_encoded['input_ids'].to(self.device),
                attention_mask=src_encoded['attention_mask'].to(self.device),
                use_cache=True,
                num_beams=num_beams,
                num_return_sequences=num_beams,
                max_length=self.max_tgt_len
            )
            # src_encoded['input_ids']: [batch_size, padding_length]
            # [batch_size, num_beams, -1]
            gen_outputs = torch.reshape(gen_outputs,(src_encoded['input_ids'].shape[0],num_beams,-1))
            # gen_outputs = [p.cpu().numpy() for p in gen_outputs]

        return gen_outputs, relation_clf_outputs, entity_clf_outputs

