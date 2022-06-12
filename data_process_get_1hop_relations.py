import json
import pyodbc
from nltk.tokenize import word_tokenize


IGONORED_DOMAIN_LIST = ['type', 'common', 'kg', 'dataworld', 'freebase', 'user']

def _tokenize_relation(r):
    return r.replace('.', ' ').replace('_', ' ').split()

# connection for freebase
odbc_conn = None
def initialize_odbc_connection():
    global odbc_conn
    # odbc_conn = pyodbc.connect(r'DRIVER=/data/virtuoso/virtuoso-opensource/lib/virtodbc.so'
    #                       r';HOST=114.212.190.19:1111'
    #                       r';UID=dba'
    #                       r';PWD=dba'
    #                       )
    odbc_conn = pyodbc.connect(
        'DRIVER=/home3/xwu/virtuoso/virtodbc.so;Host=localhost:1111;UID=dba;PWD=dba'
    )
    odbc_conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
    odbc_conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
    odbc_conn.setencoding(encoding='utf8')
    print('Freebase Virtuoso ODBC connected')

def get_out_relations_with_odbc(entity):
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    out_relations = set()

    query2 = ("""SPARQL
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
              ':' + entity + ' ?x0 ?x1 . '
                             """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    }
    """)
    # print(query2)

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query2)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query2}")
        exit(0)
    

    for row in rows:
        out_relations.add(row[0].replace('http://rdf.freebase.com/ns/', ''))

    return out_relations


def get_in_relations_with_odbc(entity):
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    in_relations = set()

    query1 = ("""SPARQL
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x0 AS ?value) WHERE {
            SELECT DISTINCT ?x0  WHERE {
            """
              '?x1 ?x0 ' + ':' + entity + '. '
                                          """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     }
     """)
    # print(query1)


    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query1)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query1}")
        exit(0)
    

    for row in rows:
        in_relations.add(row[0].replace('http://rdf.freebase.com/ns/', ''))

    return in_relations


def get_1hop_relations_with_odbc(entity):
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    relations = set()

    query = ("""SPARQL
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x0 AS ?value) WHERE {
            SELECT DISTINCT ?x0  WHERE {
            """
              '{ ?x1 ?x0 ' + ':' + entity + ' }'
              + ' UNION '
              + '{ :' + entity + ' ?x0 ?x1 ' + '}'
                                          """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     }
     """)


    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query}")
        exit(0)
    

    for row in rows:
        relations.add(row[0].replace('http://rdf.freebase.com/ns/', ''))

    return relations




def load_json(fname, mode="r", encoding="utf8"):
    if "b" in mode:
        encoding = None
    with open(fname, mode=mode, encoding=encoding) as f:
        return json.load(f)


def dump_json(obj, fname, indent=4, mode='w' ,encoding="utf8", ensure_ascii=False):
    if "b" in mode:
        encoding = None
    with open(fname, "w", encoding=encoding) as f:
        return json.dump(obj, f, indent=indent, ensure_ascii=ensure_ascii)


def get_all_candidate_entities():
    train_data = load_json('CWQ_train_all_data.json')
    dev_data = load_json('CWQ_dev_all_data.json')
    test_data = load_json('CWQ_test_all_data.json')

    unique_entities = set()
    for data in train_data:
        for ent in data['cand_entity_list']:
            unique_entities.add(ent["id"])
    
    for data in dev_data:
        for ent in data['cand_entity_list']:
            unique_entities.add(ent["id"])
    
    for data in test_data:
        for ent in data['cand_entity_list']:
            unique_entities.add(ent["id"])
    
    dump_json(list(unique_entities), 'CWQ_unique_candidate_entities.json')


def get_entities_in_out_relations():
    entities = load_json('CWQ_unique_candidate_entities.json')
    new_res = dict()
    count = 0
    for ent in entities:
        print(count)
        count += 1
        out_relations = get_out_relations_with_odbc(ent)
        out_relations = [x for x in out_relations if x.split('.')[0] not in IGONORED_DOMAIN_LIST]
        in_relations = get_in_relations_with_odbc(ent)
        in_relations = [x for x in in_relations if x.split('.')[0] not in IGONORED_DOMAIN_LIST]
        new_res[ent] = {
            'in_relations': in_relations,
            'out_relations': out_relations
        }
    
    dump_json(new_res, 'CWQ_candidate_entities_in_out_relations.json')


def sort_entities_relations_by_question(question, in_relations, out_relations):
    question_tokens = word_tokenize(question.lower())

    def key_func(r):
        r_tokens = _tokenize_relation(r)
        overlapping_val = len(set(question_tokens) & set(r_tokens))
        return overlapping_val
    
    in_relations = sorted(in_relations, key=lambda x: key_func(x), reverse=True)
    out_relations = sorted(out_relations, key=lambda x: key_func(x), reverse=True)

    return in_relations, out_relations


def filter_relations_by_domain():
    in_out_relations_map = load_json('CWQ_candidate_entities_in_out_relations.json')
    new_map = dict()
    for id in in_out_relations_map:
        in_relations = [x for x in in_out_relations_map[id]['in_relations'] if x.split('.')[0] not in IGONORED_DOMAIN_LIST]
        out_relations = [x for x in in_out_relations_map[id]['out_relations'] if x.split('.')[0] not in IGONORED_DOMAIN_LIST]
        new_map[id] = {
            'in_relations': in_relations,
            'out_relations': out_relations
        }
    dump_json(new_map, 'CWQ_candidate_entities_in_out_relations_new.json')



def add_entity_relations_to_merged_data(split, topK=3):
    merged_data = load_json('CWQ_{}_all_data.json'.format(split))
    in_out_relations_map = load_json('CWQ_candidate_entities_in_out_relations.json')
    new_data = []
    for data in merged_data:
        question = data["question"]
        for idx in range(len(data["cand_entity_list"])):
            cand_entity = data["cand_entity_list"][idx]
            in_relations, out_relations = in_out_relations_map[cand_entity["id"]]["in_relations"], in_out_relations_map[cand_entity["id"]]["out_relations"]
            filtered_in_relations, filtered_out_relations = sort_entities_relations_by_question(question, in_relations, out_relations)
            cand_entity["in_relations"] = filtered_in_relations[:topK]
            cand_entity["out_relations"] = filtered_out_relations[:topK]
            data["cand_entity_list"][idx] = cand_entity
        new_data.append(data)
    
    dump_json(new_data, 'CWQ_{}_all_data_entity_relations.json'.format(split))


def get_entities_1hop_relations():
    entities = load_json('CWQ_unique_candidate_entities.json')
    new_res = dict()
    count = 0
    for ent in entities:
        print(count)
        count += 1
        relations = get_1hop_relations_with_odbc(ent)
        relations = [x for x in relations if x.split('.')[0] not in IGONORED_DOMAIN_LIST]
        new_res[ent] = {
            '1hop_relations': relations
        }
    
    dump_json(new_res, 'CWQ_candidate_entities_1hop_relations.json')


def add_1hop_relations_to_merged_data(split):
    merged_data = load_json('CWQ_{}_all_data_entity_relations.json'.format(split))
    onehop_relations_map = load_json('CWQ_candidate_entities_1hop_relations.json')
    new_data = []
    for data in merged_data:
        for idx in range(len(data["cand_entity_list"])):
            cand_entity = data["cand_entity_list"][idx]
            onehop_relations = onehop_relations_map[cand_entity['id']]['1hop_relations']
            cand_entity["1hop_relations"] = onehop_relations
            data["cand_entity_list"][idx] = cand_entity
        new_data.append(data)
    
    dump_json(new_data, 'CWQ_{}_all_data_entity_relations_1hop.json'.format(split))



if __name__ == "__main__":
    # get_all_candidate_entities()
    # get_entities_in_out_relations()
    for split in ['train', 'dev', 'test']:
        print(split)
        # add_entity_relations_to_merged_data(split)
        add_1hop_relations_to_merged_data(split)
    # filter_relations_by_domain()
    # rels = get_1hop_relations_with_odbc('m.0dzct')
    # print(rels)
    # get_entities_1hop_relations()