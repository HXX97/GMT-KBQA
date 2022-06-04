"""
Please setup virtuoso on your machine, as well as pyodbc or SPARQLWrapper;
And substitute all "TODO:" with your own setting
"""
import pyodbc
# from SPARQLWrapper import SPARQLWrapper, JSON
import json
import time
from collections import defaultdict

odbc_conn = None
def initialize_odbc_connection():
    global odbc_conn
    # TODO:
    odbc_conn = pyodbc.connect(
        'DRIVER=/data/virtuoso/virtuoso-opensource/lib/virtodbc.so;Host=localhost:1111;UID=dba;PWD=dba'
    )
    odbc_conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
    odbc_conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
    odbc_conn.setencoding(encoding='utf8')
    print('Virtuoso ODBC connected')


def get_relations_with_odbc(data_path, limit=100):
    """Get all relations of Freebase"""
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    # {{ }}: to escape
    if limit > 0:
        query = """
        SPARQL SELECT DISTINCT ?p (COUNT(?p) as ?freq) WHERE {{
            ?subject ?p ?object
        }}
        LIMIT {}
        """.format(limit)
    else:
        query = """
        SPARQL SELECT DISTINCT ?p (COUNT(?p) as ?freq) WHERE {{
            ?subject ?p ?object
        }}
        """
    print('query: {}'.format(query))
    
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query}")
        exit(0)
    
    rtn = []
    for row in rows:
        rtn.append([row[0], int(row[1])])
    
    if len(rtn) != 0:
        write_json(data_path, rtn)


def get_entities_with_odbc(data_path, limit=100):
    """Get all entities in Freebase"""
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    # {{ }}: to escape
    if limit > 0:
        query = """
        SPARQL SELECT DISTINCT ?subject WHERE {{ 
            ?subject ?p ?object
        }} 
        LIMIT {}
        """.format(limit)
    else:
        query = """
        SPARQL SELECT DISTINCT ?subject WHERE {{ 
            ?subject ?p ?object
        }} 
        """
    print('query: {}'.format(query))
    
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query}")
        exit(0)
    
    rtn = []
    for row in rows:
        rtn.append(row[0])
    
    if len(rtn) != 0:
        write_json(data_path, rtn)

def query_relation_domain_range_odbc(input_path, output_path):
    """
    output: relation: {domain: , range:, label:}
    """
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    with open(input_path, 'r') as f:
        relations = json.load(f)
    
    res_dict = dict()
    for relation in relations:
        query = """
        SPARQL DESCRIBE {}
        """.format('<' + relation + '>')
        print('query: {}'.format(query))
        
        try:
            with odbc_conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        except Exception:
            print(f"Query Execution Failed:{query}")
            exit(0)
        
        res_dict[relation] = dict()
        for row in rows:
            if '#domain' in row[1]:
                res_dict[relation]["domain"] = row[2]
            elif '#range' in row[1]:
                res_dict[relation]["range"] = row[2]
            elif '#label' in row[1]:
                res_dict[relation]["label"] = row[2]

        print(res_dict[relation])
    
    with open(output_path, 'w') as f:
        json.dump(res_dict, fp=f, indent=4)


def query_entity_type_with_odbc(entities_path, output_path):
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    res_dict = defaultdict(list)
    entities = read_json(entities_path)
    count = 0
    for entity in entities:
        query = """
        SPARQL DESCRIBE {}
        """.format('<' + entity + '>')
        print('count: {}'.format(count))
        count += 1
        
        try:
            with odbc_conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
            for row in rows:
                if row[1] == 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type':
                    if row[2].startswith('http://dbpedia.org/ontology/'):
                        res_dict[entity].append(row[2])
        except Exception:
            print(f"Query Execution Failed:{query}")
            # exit(0)
    
    write_json(output_path, res_dict)


def write_json(json_path, data):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def post_process(input_path, output_path):
    with open(input_path, 'r') as f:
        input_data = json.load(f)
    print('load')
    output_data = list(filter(lambda item: item.startswith("http://rdf.freebase.com/ns/"), input_data))
    print('filter')
    output_data = list(map(lambda item: item.replace('http://rdf.freebase.com/ns/', ''), output_data))
    print('replace header')
    output_data = list(set(output_data))
    print('remove redundant')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)


def freebase_query_entity_type_with_odbc(entities_path, output_path):
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    res_dict = defaultdict(list)
    entities = read_json(entities_path)
    count = 0
    for entity in entities:
        query = """
        SPARQL DESCRIBE {}
        """.format('ns:' + entity)
        print('count: {}'.format(count))
        count += 1
        
        try:
            with odbc_conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
            for row in rows:
                if row[1] == 'http://rdf.freebase.com/ns/kg.object_profile.prominent_type':
                    if row[2].startswith('http://rdf.freebase.com/ns/'):
                        # res_dict[entity].append(row[2])
                        res_dict[entity].append(row[2].replace('http://rdf.freebase.com/ns/', ''))
        except Exception:
            print(f"Query Execution Failed:{query}")
            # exit(0)
    
    write_json(output_path, res_dict)

"""
*Not included in the paper!*
DBPedia related
"""

def filter_relations_by_freq(input_path, output_path, minimum_freq=5):
    input_data = read_json(input_path)
    res = []
    for item in input_data:
        rel = item[0]
        freq = item[1]
        if (rel.startswith('http://dbpedia.org/property/') or rel.startswith('http://dbpedia.org/ontology/')) and freq > minimum_freq:
            res.append(rel.replace('http://dbpedia.org/property/', 'dbp:').replace('http://dbpedia.org/ontology/', 'dbo:'))
    write_json(output_path, res)


def filter_DBPedia_relations(input_path, output_path_1, output_path_2):
    input_data = read_json(input_path)
    output1 = []
    output2 = []
    for rel in input_data:
        if rel.startswith('http://dbpedia.org/property/') or rel.startswith('http://dbpedia.org/ontology/'):
            output1.append(rel)
        else:
            output2.append(rel)
    write_json(output_path_1, output1)
    write_json(output_path_2, output2)


def query_one_hop_relations_with_odbc(entities_path, output_path):
    """query one hop relations of each entity"""
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    res_dict = defaultdict(list)
    entities = read_json(entities_path)
    count = 0

    for entity in entities:
        query = """
        SPARQL SELECT DISTINCT ?p where {{
            {{ ?s ?p {} }}
            UNION
            {{ {} ?p ?o}}
        }}
        """.format('<' + entity + '>', '<' + entity + '>')
        # print(query)
        print('count: {}'.format(count))
        count += 1
    
        try:
            with odbc_conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
            for row in rows:
                # For dbpedia: relations should starts with dbp or dbo
                if row[0].startswith('http://dbpedia.org/property/') or row[0].startswith('http://dbpedia.org/ontology/'):
                    res_dict[entity].append(row[0].replace('http://dbpedia.org/property/', 'dbp:').replace('http://dbpedia.org/ontology/', 'dbo:'))
        except Exception:
            print(f"Query Execution Failed:{query}")
            # exit(0)
    write_json(output_path, res_dict)


if __name__ == '__main__':
    time_start=time.time()
    # get_relations_with_odbc('data/DBPedia_1610_relation_all_new.json', limit=0)

    # filter_relations_by_freq('data/DBPedia_1604_relation_all_new.json', 'data/DBPedia_1610_relation_freq5.json')

    # data = read_json('data/relation_all.json')
    # print(len(data))

    # post_process('data/relation_all.json', 'data/relation_filtered.json')

    # query_relation_domain_range_odbc('data/DBPedia_1604_relations_filtered.json', 'data/DBPedia_1604_roles_all.json')
    # get_entities_with_odbc('data/entities_all.json', limit=0)

    # get_DBPedia_1604_relations_with_sparqlWrapper()

    # filter_DBPedia_relations('data/DBPedia_1604_relation_all.json','data/DBPedia_1604_relation_property_ontology.json', 'data/DBPedia_1604_relation_others.json')
    # query_entity_type_with_odbc('data/candEntities.falcon.json', 'data/candEntities.falcon.withType.json')

    # query_one_hop_relations_with_odbc('data/candEntities.falcon.json', 'data/oneHopRelations.candEntities.falcon.json')

    # candEntities = read_json('data/candEntities.falcon.json')
    # oneHopMap = read_json('data/oneHopRelations.candEntities.falcon.json')
    # for item in candEntities:
    #     if item not in oneHopMap:
    #         print(item)

    freebase_query_entity_type_with_odbc(
        'data/CWQ/entity_linking/unique_candEntities.json',
        'data/CWQ/entity_linking/CWQ_entity_prominentType_map.json'
    )

    time_end=time.time()
    print('time cost: {}'.format(time_end - time_start))
