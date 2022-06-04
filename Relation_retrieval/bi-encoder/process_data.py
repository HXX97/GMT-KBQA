import json
import random
import csv
import pandas as pd
from collections import defaultdict
"""
Data processing for Bi-encoder
create training data(.tsv) format for Bi-encoder
"""

BLANK_TOKEN = '[BLANK]'
    

def serialize_rich_relation(relation, domain_range_dict, seperator="|"):
    """
    Enrich relation representation in following format
        :relation|label|domain|range
    label, domain, range can be found in domain_range_dict
    domain_range_dict: {relation: {'domain': , 'range': , 'label':}}
    """
    if relation not in domain_range_dict:
        return relation
    else:
        res = relation
        if 'label' in domain_range_dict[relation]:
            # when label and relation are similar，do not append label
            if relation.replace('dbp:', '').replace('dbo:', '').lower() != domain_range_dict[relation]['label'].lower().replace(' ', ''):
                res += (seperator + domain_range_dict[relation]['label'])
        if 'domain' in domain_range_dict[relation]:
            res += (seperator + domain_range_dict[relation]['domain'])
        if 'range' in domain_range_dict[relation]:
            res += (seperator + domain_range_dict[relation]['range'])
        return res


def sample_data(golden_file, relations_file, output_path, sample_size=100):
    """
    get training/testing data for bi-encoder
    for each golden relation, get (sample_size -1) negative relations randomly

    golden_file: dataset with golden relations extracted
    relations_file: all relations in freebase
    """
    print(output_path)
    golden_NLQ_relations = dict()
    all_relations = read_json(relations_file)
    with open(golden_file, 'r', encoding='utf-8') as f:
        items = json.load(f)
        for item in items:
            golden_NLQ_relations[item["question"]] = list(item["gold_relation_map"].keys())

    samples = []
    for question in golden_NLQ_relations:  
        relations = golden_NLQ_relations[question]
        diff_rels = list(set(all_relations) - set(relations))
        
        negative_rels = random.sample(diff_rels, (sample_size-1) * len(relations))
        # make sure each batch contains 1 positive relation
        for idx in range(len(relations)):
            sample = []
            sample.append([question, relations[idx], '1'])
            for n_rel in negative_rels[idx * (sample_size-1): (idx+1) * (sample_size-1)]:
                sample.append([question, n_rel, '0'])
            random.shuffle(sample)
            samples.extend(sample)
        
    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1


def sample_data_rich_relation(
    golden_file, 
    relations_file, 
    relation_rich_map_path,
    output_path, 
    sample_size=100
):
    """
    sample_data() create training/test dataset based relations
    sample_data_rich_relation() create training/test dataset based on enriched relations, i.e., relation|label|domain|range
    """
    golden_NLQ_richs = dict()
    relation_rich_map = read_json(relation_rich_map_path)
    all_relations = read_json(relations_file)
    all_rich = list(set(map(lambda item: relation_rich_map[item], all_relations)))
    with open(golden_file, 'r', encoding='utf-8') as f:
        items = json.load(f)
        for item in items:
            # For LC1
            golden_NLQ_richs[item["question"].lower()] = list(set(map(
                lambda item: relation_rich_map[item], list(item["gold_relation_map"].keys())
            )))

    samples = []
    for question in golden_NLQ_richs: 
        rich = golden_NLQ_richs[question]
        diff_rich = list(set(all_rich) - set(rich))
        
        negative_rich = random.sample(diff_rich, (sample_size-1) * len(rich))
        # Make sure each batch contains 1 golden relation
        for idx in range(len(rich)):
            sample = []
            sample.append([question, rich[idx], '1'])
            for n_lab in negative_rich[idx * (sample_size-1): (idx+1) * (sample_size-1)]:
                sample.append([question, n_lab, '0'])
            random.shuffle(sample)
            samples.extend(sample)
        
    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1


def WebQSP_sample_data_1parse(
    golden_file, 
    relations_file, 
    relation_rich_map_file, 
    output_path, 
    sample_size=100
):
    """
    illustrated how we choose the golden SPARQL when multiple SPARQLs are provided
    with multiple parses:
        - SExpr_execute_right = True
        - with shortest SPARQL length
    """
    NLQ_relations_map = defaultdict(dict)
    all_relations = read_json(relations_file)
    relation_rich_map = read_json(relation_rich_map_file)
    with open(golden_file, 'r', encoding='utf-8') as f:
        items = json.load(f)
        for item in items:
            if 'train' in golden_file and len(item["goldenRelations"]) == 0:
                continue
            other_relations = set([rel for parse in item["Parses"] for rel in parse["relations"]]) - set(item["goldenRelations"])

            NLQ_relations_map[item["ProcessedQuestion"]] = {
                'goldenRelations': item["goldenRelations"],
                'otherRelations': list(other_relations)
            }
    samples = []
    for question in NLQ_relations_map:
        goldenRelations = NLQ_relations_map[question]["goldenRelations"]
        otherRelations = NLQ_relations_map[question]['otherRelations']
        # otherRelations should not be negative sample
        diff_rels = list(set(all_relations) - set(goldenRelations) - set(otherRelations))
        negative_rels = random.sample(diff_rels, (sample_size-1) * len(goldenRelations))
        
        # make sure each batch contains one golden relation
        for idx in range(len(goldenRelations)):
            sample = []
            sample.append([question, relation_rich_map[goldenRelations[idx]] if goldenRelations[idx] in relation_rich_map else goldenRelations[idx], '1'])
            for n_rel in negative_rels[idx * (sample_size-1): (idx+1) * (sample_size-1)]:
                sample.append([question, relation_rich_map[n_rel], '0'])
            random.shuffle(sample)
            samples.extend(sample)
    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1


def sample_data_mask_entity_mention(golden_file, entity_linking_file, all_relations_file, output_path, sample_size=100):
    """
    compared to sample_data()
    this method mask entity mentions in question accroding to entity linking results
    """
    print(output_path)
    golden_NLQ_relations = dict()
    all_relations = read_json(all_relations_file)
    entity_linking_res = read_json(entity_linking_file)
    with open(golden_file, 'r', encoding='utf-8') as f:
        items = json.load(f)
        for item in items:
            # mask entity mention in question
            question = item["question"].lower()
            qid = item["ID"]
            el_result = entity_linking_res[qid] if qid in entity_linking_res else []
            for eid in el_result:
                mention = el_result[eid]["mention"]
                question = question.replace(mention, BLANK_TOKEN)
            golden_NLQ_relations[question] = list(item["gold_relation_map"].keys())

    samples = []
    for question in golden_NLQ_relations: 
        relations = golden_NLQ_relations[question]
        diff_rels = list(set(all_relations) - set(relations))
        
        negative_rels = random.sample(diff_rels, (sample_size-1) * len(relations))
        # Make sure each batch contains 1 golden relation
        for idx in range(len(relations)):
            sample = []
            sample.append([question, relations[idx], '1'])
            for n_rel in negative_rels[idx * (sample_size-1): (idx+1) * (sample_size-1)]:
                sample.append([question, n_rel, '0'])
            random.shuffle(sample)
            samples.extend(sample)
        
    with open(output_path, 'w') as f:
        header = ['id', 'question', 'relation', 'label']
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        idx = 0
        for line in samples:
            writer.writerow([str(idx)] + line)
            idx += 1


def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def write_json(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)


def main():
    # get CWQ training data
    # sample_data(
    #     '../../Data/CWQ/generation/merged/CWQ_test.json',
    #     '../../Data/common_data/freebase_relations_filtered.json',
    #     'data/CWQ.test.sampled.tsv'
    # )

    # get WebQSP training data with enriched relations
    # sample_data_rich_relation(
    #     '../../Data/WEBQSP/generation/merged/WebQSP_test.json',
    #     '../../Data/common_data/freebase_relations_filtered.json',
    #     '../../Data/common_data/fb_relation_rich_map.json',
    #     'data/WebQSP.test.sampled.rich.tsv'
    # )

    # get CWQ data with entity mentions in question masked
    sample_data_mask_entity_mention(
        '../../Data/CWQ/generation/merged/CWQ_test.json',
        '../../Data/CWQ/entity_retrieval/linking_results/merged_CWQ_test_linking_results.json',
        '../../Data/common_data/freebase_relations_filtered.json',
        'data/CWQ.test.sampled.masked.tsv'
    )


if __name__=='__main__':
    main()
