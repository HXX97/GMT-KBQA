from typing import List, Tuple
from SPARQLWrapper import SPARQLWrapper, JSON
import json
import urllib
from pathlib import Path
from tqdm import tqdm

# sparql = SPARQLWrapper("http://114.212.190.19:8890/sparql")
sparql = SPARQLWrapper("http://210.28.134.34:8890/sparql")
sparql.setReturnFormat(JSON)

path = str(Path(__file__).parent.absolute())

with open(path + '/../ontology/fb_roles', 'r') as f:
    contents = f.readlines()

roles = set()
for line in contents:
    fields = line.split()
    roles.add(fields[1])

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
        f'DRIVER={path}/../lib/virtodbc.so;Host=localhost:1111;UID=dba;PWD=dba'
    )
    odbc_conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
    odbc_conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
    odbc_conn.setencoding(encoding='utf8')
    print('Freebase Virtuoso ODBC connected')

# connection for dbpedia
odbc_dbpedia_conn = None
def initialize_dbpedia_odbc_connection():
    global odbc_dbpedia_conn
    odbc_dbpedia_conn = pyodbc.connect(f'DRIVER={path}/../lib/virtodbc.so'
                          ';HOST=210.28.134.34:1113'
                          ';UID=dba'
                          ';PWD=dba'
                          )
    odbc_dbpedia_conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
    odbc_dbpedia_conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
    odbc_dbpedia_conn.setencoding(encoding='utf8')
    print('DBpedia Virtuoso ODBC connected')



def execute_query(query: str) -> List[str]:
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        # exit(0)
    rtn = []
    for result in results['results']['bindings']:
        assert len(result) == 1  # only select one variable
        for var in result:
            rtn.append(result[var]['value'].replace('http://rdf.freebase.com/ns/', '').replace("-08:00", ''))

    return rtn

def execute_query_with_odbc(query:str) -> List[str]:
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    # print('successfully connnected to Freebase ODBC')
    result_set = set()
    query2 = "SPARQL "+query
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query2)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query2}")
        exit(0)
    
    for row in rows:
        result_set.add(row[0])

    return result_set

def execute_query_with_odbc_filter_answer(query:str) -> List[str]:
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    result_set = set()

    query2 = "SPARQL "+query
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query2)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query2}")
        exit(0)
    
    for row in rows:
        result_set.add(row[0].replace('http://rdf.freebase.com/ns/', '').replace("-08:00", ''))

    return result_set



def execute_unary(type: str) -> List[str]:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
             '?x0 :type.object.type :' + type + '. '
                                                """
    }
    }
    """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        rtn.append(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return rtn


def execute_binary(relation: str) -> List[Tuple[str, str]]:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT DISTINCT ?x0 ?x1 WHERE {
    """
             '?x0 :' + relation + ' ?x1. '
                                  """
    }
    """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        rtn.append((result['x0']['value'], result['x1']['value']))

    return rtn


def get_types(entity: str) -> List[str]:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
             ':' + entity + ' :type.object.type ?x0 . '
                            """
    }
    }
    """)
    # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        rtn.append(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return rtn


def get_types_with_odbc(entity: str)  -> List[str]:

    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    types = set()

    query = ("""SPARQL
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
             ':' + entity + ' :type.object.type ?x0 . '
                            """
    }
    }
    """)

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query}")
        # exit(0)
    

    for row in rows:
        types.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
    
    if len(types)==0:
        return []
    else:
        return list(types)


def get_notable_type(entity: str):
    query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        
        """
             ':' + entity + ' :common.topic.notable_types ?y . '
                            """
        ?y :type.object.name ?x0
        FILTER (lang(?x0) = 'en')
    }
    }
    """)

    # print(query)
    sparql.setQuery(query)
    results = sparql.query().convert()
    rtn = []
    for result in results['results']['bindings']:
        rtn.append(result['value']['value'])

    if len(rtn) == 0:
        rtn = ['entity']

    return rtn


def get_friendly_name(entity: str) -> str:
    query = ("""
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX : <http://rdf.freebase.com/ns/> 
    SELECT (?x0 AS ?value) WHERE {
    SELECT DISTINCT ?x0  WHERE {
    """
             ':' + entity + ' :type.object.name ?x0 . '
                            """
    }
    }
    """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        if result['value']['xml:lang'] == 'en':
            rtn.append(result['value']['value'])

    if len(rtn) == 0:
        query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
                 ':' + entity + ' :common.topic.alias ?x0 . '
                                """
        }
        }
        """)
        # # print(query)
        sparql.setQuery(query)
        try:
            results = sparql.query().convert()
        except urllib.error.URLError:
            print(query)
            exit(0)
        for result in results['results']['bindings']:
            if result['value']['xml:lang'] == 'en':
                rtn.append(result['value']['value'])

    if len(rtn) == 0:
        return 'null'

    return rtn[0]


def get_degree(entity: str):
    degree = 0

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT count(?x0) as ?value WHERE {
            """
              '?x1 ?x0 ' + ':' + entity + '. '
                                          """
     FILTER regex(?x0, "http://rdf.freebase.com/ns/")
     }
     """)
    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        degree += int(result['value']['value'])

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT count(?x0) as ?value WHERE {
        """
              ':' + entity + ' ?x0 ?x1 . '
                             """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    """)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        degree += int(result['value']['value'])

    return degree


def get_in_attributes(value: str):
    in_attributes = set()

    query1 = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/> 
                SELECT (?x0 AS ?value) WHERE {
                SELECT DISTINCT ?x0  WHERE {
                """
              '?x1 ?x0 ' + value + '. '
                                   """
    FILTER regex(?x0, "http://rdf.freebase.com/ns/")
    }
    }
    """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        in_attributes.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return in_attributes


def get_in_relations(entity: str):
    in_relations = set()

    query1 = ("""
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

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        in_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return in_relations


def get_in_entities(entity: str, relation: str):
    neighbors = set()

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x1 AS ?value) WHERE {
            SELECT DISTINCT ?x1  WHERE {
            """
              '?x1' + ':' + relation + ':' + entity + '. '
                                                      """
                 FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                 }
                 }
                 """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return neighbors


def get_in_entities_for_literal(value: str, relation: str):
    neighbors = set()

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT (?x1 AS ?value) WHERE {
            SELECT DISTINCT ?x1  WHERE {
            """
              '?x1' + ':' + relation + ' ' + value + '. '
                                                     """
                FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                }
                }
                """)
    # print(query1)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    for result in results['results']['bindings']:
        neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return neighbors


def get_out_relations(entity: str):
    out_relations = set()

    query2 = ("""
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

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        out_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return out_relations


def get_out_entities(entity: str, relation: str):
    neighbors = set()

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x1 AS ?value) WHERE {
        SELECT DISTINCT ?x1  WHERE {
        """
              ':' + entity + ':' + relation + ' ?x1 . '
                                              """
                     FILTER regex(?x1, "http://rdf.freebase.com/ns/")
                     }
                     }
                     """)
    # print(query2)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return neighbors


def get_entities_cmp(value, relation: str, cmp: str):
    neighbors = set()

    query2 = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x1 AS ?value) WHERE {
        SELECT DISTINCT ?x1  WHERE {
        """
              '?x1' + ':' + relation + ' ?sk0 . '
                                       """
              FILTER regex(?x1, "http://rdf.freebase.com/ns/")
              """
                                       f'FILTER (?sk0 {cmp} {value})'
                                       """
                                       }
                                       }
                                       """)
    # print(query2)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    for result in results['results']['bindings']:
        neighbors.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return neighbors


def get_adjacent_relations(entity: str):
    in_relations = set()
    out_relations = set()

    query1 = ("""
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

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    except Exception:
        print(query1)
        results = None
    if results:
        for result in results['results']['bindings']:
            in_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    query2 = ("""
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

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        results = None
        # exit(0)
    except Exception:
        print(query2)
        results = None

    if results: 
        for result in results['results']['bindings']:
            out_relations.add(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

    return in_relations, out_relations

def get_adjacent_relations_with_odbc(entity:str):
    in_relations = set()
    out_relations = set()
    

    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()

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
        rows=[]
        # exit(0)


    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        # r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        in_relations.add(r0)
        

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
    
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query2)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query2}")
        rows = []
        # exit(0)
    
    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        
        out_relations.add(r0)
        

    return in_relations, out_relations
    

def get_2hop_relations_from_2entities(entity0: str, entity1: str):  # m.027lnzs  m.0zd6  3200017000000
    query = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/>
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
             '?x1 ?x0 ' + ':' + entity0 + ' .\n' + '?x1 ?y ' + ':' + entity1 + ' .'
                                                                               """
                                                       FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                                                       FILTER regex(?y, "http://rdf.freebase.com/ns/")
                                                       }
            """)
    # print(query)
    pass


def get_2hop_relations_with_odbc(entity: str):
    in_relations = set()
    out_relations = set()
    paths = []

    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()


    query1 = ("""SPARQL 
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX ns: <http://rdf.freebase.com/ns/>
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
              '?x1 ?x0 ' + 'ns:' + entity + '. '
                                          """
                ?x2 ?y ?x1 .
                  FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
                  FILTER (?y != rdf:type && ?y != rdfs:label)
                  FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
                  FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
                  FILTER( !regex(?x0,"wikipedia","i"))
                  FILTER( !regex(?y,"wikipedia","i"))
                  FILTER( !regex(?x0,"type.object","i"))
                  FILTER( !regex(?y,"type.object","i"))
                  FILTER( !regex(?x0,"common.topic","i"))
                  FILTER( !regex(?y,"common.topic","i"))
                  FILTER( !regex(?x0,"_id","i"))
                  FILTER( !regex(?y,"_id","i"))
                  FILTER( !regex(?x0,"#type","i"))
                  FILTER( !regex(?y,"#type","i"))
                  FILTER( !regex(?x0,"#label","i"))
                  FILTER( !regex(?y,"#label","i"))
                  FILTER( !regex(?x0,"/ns/freebase","i"))
                  FILTER( !regex(?y,"/ns/freebase","i"))
                  FILTER( !regex(?x0, "ns/common."))
                  FILTER( !regex(?y, "ns/common."))
                  FILTER( !regex(?x0, "ns/type."))
                  FILTER( !regex(?y, "ns/type."))
                  FILTER( !regex(?x0, "ns/kg."))
                  FILTER( !regex(?y, "ns/kg."))
                  FILTER( !regex(?x0, "ns/user."))
                  FILTER( !regex(?y, "ns/user."))
                  FILTER( !regex(?x0, "ns/dataworld."))
                  FILTER( !regex(?y, "ns/dataworld."))
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  LIMIT 1000
                  """)
    # print(query1)
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query1)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query1}")
        rows=[]
        # exit(0)


    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        in_relations.add(r0)
        in_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0, r1))
        

    query2 = ("""SPARQL 
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX ns: <http://rdf.freebase.com/ns/> 
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
              '?x1 ?x0 ' + 'ns:' + entity + '. '
                                          """
                ?x1 ?y ?x2 .
                """
                  'FILTER (?x2 != ns:'+entity+' )'
                  """
                  FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
                  FILTER (?y != rdf:type && ?y != rdfs:label)
                  FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
                  FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
                  FILTER( !regex(?x0,"wikipedia","i"))
                  FILTER( !regex(?y,"wikipedia","i"))
                  FILTER( !regex(?x0,"type.object","i"))
                  FILTER( !regex(?y,"type.object","i"))
                  FILTER( !regex(?x0,"common.topic","i"))
                  FILTER( !regex(?y,"common.topic","i"))
                  FILTER( !regex(?x0,"_id","i"))
                  FILTER( !regex(?y,"_id","i"))
                  FILTER( !regex(?x0,"#type","i"))
                  FILTER( !regex(?y,"#type","i"))
                  FILTER( !regex(?x0,"#label","i"))
                  FILTER( !regex(?y,"#label","i"))
                  FILTER( !regex(?x0,"/ns/freebase","i"))
                  FILTER( !regex(?y,"/ns/freebase","i"))
                  FILTER( !regex(?x0, "ns/common."))
                  FILTER( !regex(?y, "ns/common."))
                  FILTER( !regex(?x0, "ns/type."))
                  FILTER( !regex(?y, "ns/type."))
                  FILTER( !regex(?x0, "ns/kg."))
                  FILTER( !regex(?y, "ns/kg."))
                  FILTER( !regex(?x0, "ns/user."))
                  FILTER( !regex(?y, "ns/user."))
                  FILTER( !regex(?x0, "ns/dataworld."))
                  FILTER( !regex(?y, "ns/dataworld."))
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  LIMIT 1000
                  """)

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query2)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query2}")
        rows = []
        # exit(0)
    
    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        in_relations.add(r0)
        out_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0, r1 + '#R'))

    
    query3 = ("""SPARQL 
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                """
              'ns:' + entity + ' ?x0 ?x1 . '
                             """
                ?x2 ?y ?x1 .
                  FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
                  FILTER (?y != rdf:type && ?y != rdfs:label)
                  FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
                  FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
                  FILTER( !regex(?x0,"wikipedia","i"))
                  FILTER( !regex(?y,"wikipedia","i"))
                  FILTER( !regex(?x0,"type.object","i"))
                  FILTER( !regex(?y,"type.object","i"))
                  FILTER( !regex(?x0,"common.topic","i"))
                  FILTER( !regex(?y,"common.topic","i"))
                  FILTER( !regex(?x0,"_id","i"))
                  FILTER( !regex(?y,"_id","i"))
                  FILTER( !regex(?x0,"#type","i"))
                  FILTER( !regex(?y,"#type","i"))
                  FILTER( !regex(?x0,"#label","i"))
                  FILTER( !regex(?y,"#label","i"))
                  FILTER( !regex(?x0,"/ns/freebase","i"))
                  FILTER( !regex(?y,"/ns/freebase","i"))
                  FILTER( !regex(?x0, "ns/common."))
                  FILTER( !regex(?y, "ns/common."))
                  FILTER( !regex(?x0, "ns/type."))
                  FILTER( !regex(?y, "ns/type."))
                  FILTER( !regex(?x0, "ns/kg."))
                  FILTER( !regex(?y, "ns/kg."))
                  FILTER( !regex(?x0, "ns/user."))
                  FILTER( !regex(?y, "ns/user."))
                  FILTER( !regex(?x0, "ns/dataworld."))
                  FILTER( !regex(?y, "ns/dataworld."))
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  LIMIT 1000
                  """)

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query3)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query3}")
        rows = []
        # exit(0)
    
    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        out_relations.add(r0)
        in_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0 + '#R', r1))


    query4 = ("""SPARQL 
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX ns: <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                """
              'ns:' + entity + ' ?x0 ?x1 . '
                             """
                ?x1 ?y ?x2 .
                """
                  'FILTER (?x2 != ns:'+entity+' )'
                """
                FILTER (?x0 != rdf:type && ?x0 != rdfs:label)
                FILTER (?y != rdf:type && ?y != rdfs:label)
                FILTER(?x0 != ns:type.object.type && ?x0 != ns:type.object.instance)
                FILTER(?y != ns:type.object.type && ?y != ns:type.object.instance)
                FILTER( !regex(?x0,"wikipedia","i"))
                FILTER( !regex(?y,"wikipedia","i"))
                FILTER( !regex(?x0,"type.object","i"))
                FILTER( !regex(?y,"type.object","i"))
                FILTER( !regex(?x0,"common.topic","i"))
                FILTER( !regex(?y,"common.topic","i"))
                FILTER( !regex(?x0,"_id","i"))
                FILTER( !regex(?y,"_id","i"))
                FILTER( !regex(?x0,"#type","i"))
                FILTER( !regex(?y,"#type","i"))
                FILTER( !regex(?x0,"#label","i"))
                FILTER( !regex(?y,"#label","i"))
                FILTER( !regex(?x0,"/ns/freebase","i"))
                FILTER( !regex(?y,"/ns/freebase","i"))
                FILTER( !regex(?x0, "ns/common."))
                FILTER( !regex(?y, "ns/common."))
                FILTER( !regex(?x0, "ns/type."))
                FILTER( !regex(?y, "ns/type."))
                FILTER( !regex(?x0, "ns/kg."))
                FILTER( !regex(?y, "ns/kg."))
                FILTER( !regex(?x0, "ns/user."))
                FILTER( !regex(?y, "ns/user."))
                FILTER( !regex(?x0, "ns/dataworld."))
                FILTER( !regex(?y, "ns/dataworld."))
                FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                FILTER regex(?y, "http://rdf.freebase.com/ns/")
                }
                LIMIT 1000
                """)

    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query4)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query4}")
        rows = []
        # exit(0)

    for row in rows:
        r0 = row[0].replace('http://rdf.freebase.com/ns/', '')
        r1 = row[1].replace('http://rdf.freebase.com/ns/', '')
        out_relations.add(r0)
        out_relations.add(r1)

        if r0 in roles and r1 in roles:
            paths.append((r0 + '#R', r1 + '#R'))

    return in_relations, out_relations, paths


def get_2hop_relations(entity: str):
    in_relations = set()
    out_relations = set()
    paths = []

    query1 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/>
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
              '?x1 ?x0 ' + ':' + entity + '. '
                                          """
                ?x2 ?y ?x1 .
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  """)

    sparql.setQuery(query1)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query1)
        exit(0)
    except Exception:
        print(query1)
        results = None
        
    if results:
        for result in results['results']['bindings']:
            r1 = result['r1']['value'].replace('http://rdf.freebase.com/ns/', '')
            r0 = result['r0']['value'].replace('http://rdf.freebase.com/ns/', '')
            in_relations.add(r0)
            in_relations.add(r1)

            if r0 in roles and r1 in roles:
                paths.append((r0, r1))

    query2 = ("""
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX : <http://rdf.freebase.com/ns/> 
            SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
            """
              '?x1 ?x0 ' + ':' + entity + '. '
                                          """
                ?x1 ?y ?x2 .
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  """)

    sparql.setQuery(query2)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query2)
        exit(0)
    except Exception:
        print(query2)
        results = None
    if results:
        for result in results['results']['bindings']:
            r1 = result['r1']['value'].replace('http://rdf.freebase.com/ns/', '')
            r0 = result['r0']['value'].replace('http://rdf.freebase.com/ns/', '')
            out_relations.add(r1)
            in_relations.add(r0)

            if r0 in roles and r1 in roles:
                paths.append((r0, r1 + '#R'))

    query3 = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                """
              ':' + entity + ' ?x0 ?x1 . '
                             """
                ?x2 ?y ?x1 .
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  """)

    sparql.setQuery(query3)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query3)
        exit(0)
    except Exception:
        print(query3)
        results = None
    
    if results:
        for result in results['results']['bindings']:
            r1 = result['r1']['value'].replace('http://rdf.freebase.com/ns/', '')
            r0 = result['r0']['value'].replace('http://rdf.freebase.com/ns/', '')
            in_relations.add(r1)
            out_relations.add(r0)

            if r0 in roles and r1 in roles:
                paths.append((r0 + '#R', r1))

    query4 = ("""
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX : <http://rdf.freebase.com/ns/>
                SELECT distinct ?x0 as ?r0 ?y as ?r1 WHERE {
                """
              ':' + entity + ' ?x0 ?x1 . '
                             """
                ?x1 ?y ?x2 .
                  FILTER regex(?x0, "http://rdf.freebase.com/ns/")
                  FILTER regex(?y, "http://rdf.freebase.com/ns/")
                  }
                  """)

    sparql.setQuery(query4)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query4)
        exit(0)
    except Exception:
        print(query4)
        results=None
    
    if results:
        for result in results['results']['bindings']:
            r1 = result['r1']['value'].replace('http://rdf.freebase.com/ns/', '')
            r0 = result['r0']['value'].replace('http://rdf.freebase.com/ns/', '')
            out_relations.add(r1)
            out_relations.add(r0)

            if r0 in roles and r1 in roles:
                paths.append((r0 + '#R', r1 + '#R'))

    return in_relations, out_relations, paths


def get_label(entity: str) -> str:
    """Get the label of an entity in Freebase"""
    query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?label) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
             ':' + entity + ' rdfs:label ?x0 . '
                            """
                            FILTER (langMatches( lang(?x0), "EN" ) )
                             }
                             }
                             """)
    # # print(query)
    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
    except urllib.error.URLError:
        print(query)
        exit(0)
    rtn = []
    for result in results['results']['bindings']:
        label = result['label']['value']
        rtn.append(label)
    if len(rtn) != 0:
        return rtn[0]
    else:
        return None


import pyodbc
def pyodbc_test():
    conn = pyodbc.connect(r'DRIVER=/data/virtuoso/virtuoso-opensource/lib/virtodbc.so'
                          r';SERVER=210.28.134.34:1111'
                          r';UID=dba;PWD=dba'
                          )
    print(conn)
    conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf8')
    conn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
    conn.setencoding(encoding='utf8')
    
    with conn.cursor() as cursor:
        cursor.execute("SPARQL SELECT ?subject ?object WHERE { ?subject rdfs:subClassOf ?object }")
        rows = cursor.fetchall()
    
    for row in rows:
        row = str(row)
        print(row)
    conn.commit()
    conn.close()



def get_label_with_odbc(entity: str) -> str:
    """Get the label of an entity in Freebase"""

    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
        
    query = ("""SPARQL
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX ns: <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?label) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
             'ns:' + entity + ' rdfs:label ?x0 . '
                            """
                            FILTER (langMatches( lang(?x0), "EN" ) )
                             }
                             }
                             """)

    # query = query.replace("\n"," ")
    # print(query)
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query}")
        exit(0)
    
    
    rtn = []
    for row in rows:
        # print(type(row))
        rtn.append(row[0])
    
    if len(rtn) != 0:
        return rtn[0]
    else:
        return None


def get_in_relations_with_odbc(entity: str) -> str:
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
    

def get_out_relations_with_odbc(entity: str) -> str:
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


def get_wikipage_id_from_dbpedia_uri(dbpedia_uri: str):
    # build connection
    global odbc_dbpedia_conn
    if odbc_dbpedia_conn == None:
        initialize_dbpedia_odbc_connection()
    
    wiki_page_id = set()
    query = ("""SPARQL
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT DISTINCT ?x0  WHERE {
        """
              f'<{dbpedia_uri}> <http://dbpedia.org/ontology/wikiPageID> ?x0 .'
        """
    }
    """)

    try:
        with odbc_dbpedia_conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query}")
        exit(0)
    

    for row in rows:
        wiki_page_id.add(row[0])
    
    if len(wiki_page_id)==0:
        return ''
    else:
        return list(wiki_page_id)[0]


def get_freebase_mid_from_wikiID(wikiID: int):
    # build connection
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    
    mid = set()

    query2 = ("""SPARQL
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
              '?x0 <http://rdf.freebase.com/key/wikipedia.en_id> ' + f'"{wikiID}"'
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
        mid.add(row[0].replace('http://rdf.freebase.com/ns/', ''))
    
    if len(mid)==0:
        return ''
    else:
        return list(mid)[0]


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


def execuate_reduced_sparql(split):
    print(split)
    dataset = load_json('/home3/xwu/workspace/QDT2SExpr/CWQ/data/WebQSP/merged_all_data/richRelation_2hopValidation_richEntity_CrossEntropyLoss_top100_parse0_ep6/WebQSP_{}_all_data_noQdt_parse0_with_time_constraint.json'.format(split))
    new_data = []
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    for item in dataset:
        item["reduced_answer"] = []
        reduced_sparql = item["reduced_sparql"]
        query = "SPARQL " + reduced_sparql
        try:
            with odbc_conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        except Exception:
            print(f"Query Execution Failed:{query}")
            exit(0)
        
        for row in rows:
            item["reduced_answer"].append(row[0].replace('http://rdf.freebase.com/ns/',''))
        new_data.append(item)
    
    dump_json(new_data, '/home3/xwu/workspace/QDT2SExpr/CWQ/data/WebQSP/merged_all_data/richRelation_2hopValidation_richEntity_CrossEntropyLoss_top100_parse0_ep6/WebQSP_{}_all_data_noQdt_parse0_with_time_constraint.json'.format(split))


def get_entity_labels(src_path, tgt_path):
    entities_list = load_json(src_path)
    res = dict()
    # for entity in entities_list:
    for entity in tqdm(entities_list, total=len(entities_list),desc=f'Querying entity labels'):
        label = get_label_with_odbc(entity)
        res[entity] = label
    dump_json(res, tgt_path)


def query_rich_relation(relation):
    global odbc_conn
    if odbc_conn == None:
        initialize_odbc_connection()
    query = """
        SPARQL DESCRIBE {}
        """.format(relation)
    print('query: {}'.format(query))
    try:
        with odbc_conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    except Exception:
        print(f"Query Execution Failed:{query}")
        exit(0)
    for row in rows:
        print(row)
        # if '#domain' in row[1]:
        #     print(row)
        # elif '#range' in row[1]:
        #     print(row)
        # elif '#label' in row[1]:
        #     print(row)


if __name__=='__main__':
    
    # pyodbc_test()
    
    # print(get_label('m.04tfqf'))
    print(get_label_with_odbc('m.0rczx'))
    # print(get_in_relations_with_odbc('m.04tfqf'))
    # print(get_out_relations_with_odbc('m.04tfqf'))

    # print(get_label('m.0fjp3'))
    # print(get_label_with_odbc('m.0fjp3'))
    # print(get_label('m.0z33s'))
    # print(get_2hop_relations_with_odbc('m.0rv97'))

    # print(get_wikipage_id_from_dbpedia_uri("http://dbpedia.org/resource/China"))

    
    # sparql = """
    # PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    # PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    # PREFIX ns: <http://rdf.freebase.com/ns/> 

    # SELECT count(DISTINCT ?p)  WHERE {
    # ?s ?p ?o .
    # }
    # """
    # print(execute_query_with_odbc(sparql))
    
    # print(get_adjacent_relations_with_odbc('m.0rv97'))
    # print(get_label_with_odbc('m.0y80cnb'))

    # print(get_freebase_mid_from_wikiID(39027))


    # in_rel = get_in_relations('m.04tfqf')
    # print(type(in_rel))
    # print(in_rel)
    # for split in ['train', 'dev', 'test']:
    #     execuate_reduced_sparql(split)

    # get_entity_labels()
    # query_rich_relation('ns:m.04tfqf')