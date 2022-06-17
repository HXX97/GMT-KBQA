from torch.utils.data import Dataset
import torch

IGONORED_DOMAIN_LIST = ['type', 'common', 'kg', 'dataworld']


def _tokenize_relation(r):
    return r.replace('.', ' ').replace('_', ' ').split()

class MTLGenerationExample:
    """
    Multi Task Generation Example
    """
    def __init__(self, dict_data) -> None:
        """ Initialize from dict data"""
        self.ID = dict_data['ID']
        self.question = dict_data['question']
        self.comp_type = dict_data['comp_type']
        self.sprql = dict_data['sparql']
        self.sexpr = dict_data['sexpr']
        self.normed_sexpr = dict_data['normed_sexpr']
        self.gold_entity_map = dict_data['gold_entity_map']
        self.gold_relation_map = dict_data['gold_relation_map']
        self.gold_type_map = dict_data['gold_type_map']
        self.cand_relation_list = dict_data['cand_relation_list']
        self.answer = dict_data['answer']
        self.cand_entity_list = dict_data['cand_entity_list']
        # self.gold_structure = dict_data['normed_all_masked_sexpr']
        # self.cand_structure_list = dict_data['cand_structure_list']
        self.disambiguated_cand_entity = dict_data['disambiguated_cand_entity']


    def __str__(self) -> str:
        return f'{self.question}\n\t->{self.normed_sexpr}'

    def __repr__(self) -> str:
        return self.__str__()


class MTLGenDataset(Dataset):
    """Dataset for MTLGeneration"""

    def __init__(
        self, 
        examples, 
        tokenizer, 
        do_lower=True,
        normalize_relations=False,
        max_src_len=256, 
        max_tgt_len=196,
        max_structure_tgt_len=70,
        add_prefix=False,
    ):
        # super().__init__()
        self.examples = examples
        self.tokenizer = tokenizer
        self.do_lower = do_lower
        self.normalize_relations = normalize_relations
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.max_structure_tgt_len = max_structure_tgt_len
        self.add_prefix = add_prefix
        self.REL_TOKEN = ' [REL] '
        self.ENT_TOKEN = ' [ENT] '
        self.LITERAL_TOKEN = ' [LIT] '
        self.SEPERATOR = ' | '
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        
        ID = example.ID
        question = example.question
        normed_sexpr = example.normed_sexpr

        candidate_relations = [x[0] for x in example.cand_relation_list]
        candidate_rich_relations = [x[2] for x in example.cand_relation_list]
        gold_relation_set = set(example.gold_relation_map.keys())
        
        relation_labels = [(rel in gold_relation_set) for rel in candidate_relations]
        relation_clf_pairs_labels = torch.LongTensor(relation_labels)

        # entity id identifies diffrent entities
        gold_entities_ids_set = set([item.lower() for item in example.gold_entity_map.keys()])

        entity_labels = [(ent['id'] in gold_entities_ids_set) for ent in example.cand_entity_list]
        entity_clf_pairs_labels = torch.LongTensor(entity_labels)

        # gold_structure = example.gold_structure
        # candidate_structures = example.cand_structure_list
        # structure_labels = [(item == gold_structure) for item in candidate_structures]
        # structure_clf_labels = torch.LongTensor(structure_labels)

        input_src = question

        if self.do_lower:
            input_src = input_src.lower()
            normed_sexpr = normed_sexpr.lower()
        
        gen_src = input_src
        if self.add_prefix:
            gen_src = 'Translate to S-Expression: ' + input_src
        if self.do_lower:
            gen_src = gen_src.lower()
        tokenized_src = self.tokenizer(
                        gen_src,
                        max_length=self.max_src_len,
                        truncation=True,
                        return_tensors='pt',
                        #padding='max_length',
                        ).data['input_ids'].squeeze(0)
        
        # Concatenate candidate entities & relations
        gen_src_concatenated = input_src
        if self.add_prefix:
            gen_src_concatenated = 'Translate to S-Expression: ' + gen_src_concatenated
        for rel in example.cand_relation_list:
            logits = float(rel[1])
            if logits > 0.0:
                if self.normalize_relations:
                    gen_src_concatenated += self.REL_TOKEN + _textualize_relation(rel[0])
                else:
                    gen_src_concatenated += self.REL_TOKEN + rel[0]
        gen_src_concatenated += self.SEPERATOR
        
        for ent in example.disambiguated_cand_entity:
            gen_src_concatenated += self.ENT_TOKEN + ent['label']
        gen_src_concatenated += self.SEPERATOR
        # if len(candidate_structures) > 0:
        #     gen_src_concatenated += candidate_structures[0]
        
        tokenized_src_concatenated = self.tokenizer(
            gen_src_concatenated,
            max_length=self.max_src_len,
            truncation=True,
            return_tensors='pt',
        ).data['input_ids'].squeeze(0)

        # concatenate golden entities/relations
        gen_src_golden_concatenated = input_src
        if self.add_prefix:
            gen_src_golden_concatenated = 'Translate to S-Expression: ' + gen_src_golden_concatenated
        for rel in example.gold_relation_map:
            if self.normalize_relations:
                gen_src_golden_concatenated += self.REL_TOKEN + _textualize_relation(rel)
            else:
                gen_src_golden_concatenated += self.REL_TOKEN + rel
        gen_src_golden_concatenated += self.SEPERATOR
        for mid in example.gold_entity_map:
            gen_src_golden_concatenated += self.ENT_TOKEN + example.gold_entity_map[mid] # concat label
        gen_src_golden_concatenated += self.SEPERATOR
        # if len(candidate_structures) > 0:
        #     gen_src_concatenated += candidate_structures[0]
        
        tokenized_src_golden_concatenated = self.tokenizer(
            gen_src_golden_concatenated,
            max_length=self.max_src_len,
            truncation=True,
            return_tensors='pt',
        ).data['input_ids'].squeeze(0)


        with self.tokenizer.as_target_tokenizer():
            tokenized_tgt = self.tokenizer(
                normed_sexpr,
                max_length=self.max_tgt_len,
                truncation=True,
                return_tensors='pt',
                #padding='max_length',
            ).data['input_ids'].squeeze(0)
        
        # with self.tokenizer.as_target_tokenizer():
        #     tokenized_structure_gen_tgt = self.tokenizer(
        #         gold_structure,
        #         max_length=self.max_structure_tgt_len,
        #         truncation=True,
        #         return_tensors='pt',
        #     ).data['input_ids'].squeeze(0)
        
        tokenized_relation_clf_pairs = []
        
        for cand_rel in candidate_relations:
            if self.normalize_relations:
                cand_rel = _textualize_relation(cand_rel)
            
            rel_src = input_src
            if self.add_prefix:
                rel_src = 'Relation Classification: ' + rel_src
            
            if self.do_lower:
                rel_src = rel_src.lower()
                cand_rel = cand_rel.lower()

            tokenized_relation_pair = self.tokenizer(
                rel_src,
                cand_rel,
                max_length=self.max_src_len,
                truncation='longest_first',
                return_tensors='pt',
                # padding='max_length',
            ).data['input_ids'].squeeze(0)
            
            tokenized_relation_clf_pairs.append(tokenized_relation_pair)
        
        tokenized_rich_relation_clf_pairs = []

        for cand_rich_rel in candidate_rich_relations:
            rel_src = input_src
            if self.add_prefix:
                rel_src = 'Relation Classification: ' + rel_src
            
            if self.do_lower:
                rel_src = rel_src.lower()
                cand_rich_rel = cand_rich_rel.lower()
            
            tokenized_rich_relation_pair = self.tokenizer(
                rel_src,
                cand_rich_rel,
                max_length=self.max_src_len,
                truncation='longest_first',
                return_tensors='pt',
                # padding='max_length',
            ).data['input_ids'].squeeze(0)

            tokenized_rich_relation_clf_pairs.append(tokenized_rich_relation_pair)

        tokenized_entity_clf_pairs = []
        question_tokens = question.split(' ')
        
        for cand_ent in example.cand_entity_list:
            label = cand_ent['label']
            def key_func(r):
                r_tokens = _tokenize_relation(r)
                overlapping_val = len(set(question_tokens) & set(r_tokens))
                return(
                    -overlapping_val
                )
            
            one_hop_relations = cand_ent['1hop_relations']
            one_hop_relations = [x for x in one_hop_relations if x.split('.')[0] not in IGONORED_DOMAIN_LIST]
            one_hop_relations = sorted(one_hop_relations, key=lambda x: key_func(x))

            ent_info = label
            for rel in one_hop_relations[:3]:
                if self.normalize_relations:
                    ent_info += (" | " + _textualize_relation(rel))
                else:
                    ent_info += (" | " + rel)
            if self.do_lower:
                ent_info = ent_info.lower()
            
            ent_src = input_src
            if self.add_prefix:
                ent_src = 'Entity Classification: ' + input_src
            
            tokenized_entity_pair = self.tokenizer(
                ent_src,
                ent_info, 
                max_length=self.max_src_len,
                truncation='longest_first',
                return_tensors='pt'
            ).data['input_ids'].squeeze(0)

            tokenized_entity_clf_pairs.append(tokenized_entity_pair)
        
        #tokenized_structure_clf_pairs = []

        # for cand_stru in candidate_structures:
        #     stru_src = input_src
        #     if self.add_prefix:
        #         stru_src = 'Structure Classification: ' + stru_src
        #     if self.do_lower:
        #         stru_src = stru_src.lower()
        #         cand_stru = cand_stru.lower()
            
        #     tokenized_structure_pair = self.tokenizer(
        #         stru_src,
        #         cand_stru,
        #         max_length=self.max_src_len,
        #         truncation='longest_first',
        #         return_tensors='pt'
        #     ).data['input_ids'].squeeze(0)

        #     tokenized_structure_clf_pairs.append(tokenized_structure_pair)

        return (
            tokenized_src, 
            tokenized_tgt, 
            tokenized_relation_clf_pairs, 
            relation_clf_pairs_labels,
            # ID,
            [input_src],
            candidate_relations,
            tokenized_entity_clf_pairs,
            entity_clf_pairs_labels,
            example.cand_entity_list,
            #tokenized_structure_gen_tgt,
            #candidate_structures,
            #structure_clf_labels,
            tokenized_rich_relation_clf_pairs,
            candidate_rich_relations,
            #tokenized_structure_clf_pairs,
            tokenized_src_concatenated,
            tokenized_src_golden_concatenated
        )



def _textualize_relation(r):
    """return a relation string with '_' and '.' replaced"""
    if "_" in r: # replace "_" with " "
        r = r.replace("_", " ")
    if "." in r: # replace "." with " , "
        r = r.replace(".", " , ")
    return r       
        
