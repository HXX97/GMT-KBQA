from SPARQLWrapper import SPARQLWrapper, JSON
import requests
import json

sparql_util_conf = {
    'sparql_url':'http://210.28.134.34:8890/sparql',
    'entity_link_conf':{
        'key':'',
        'limit':5,
        'indent':True
    },
    'goole-api-url':'https://kgsearch.googleapis.com:443/v1/entities:search',
    'proxies':{
            "http": "http://114.212.82.29:7890",
            "https": "http://114.212.82.29:7890",
        }
}

class SparqlUtil():
    def __init__(self):
        global sparql_util_conf
        self.sparql_util_conf = sparql_util_conf
        self.sparql = SPARQLWrapper(sparql_util_conf["sparql_url"])
        self.sparql.setReturnFormat(JSON)
    
    def find_entity(self,query):
        data = dict(self.sparql_util_conf['entity_link_conf'])
        data['query'] = query
        result = requests.get(self.sparql_util_conf['goole-api-url'],params=data,proxies=self.sparql_util_conf['proxies'],verify=False)
        result = result.json()
        query_result = []
        for item in result["itemListElement"]:
            mid = item["result"]["@id"].split(":")[1][1:].replace("/",".")
            query_result.append([item["result"]["name"],mid,item["resultScore"]])
        query_result.sort(key = lambda x:float(x[2]),reverse=True)
        return query_result
    
    def query_entity_label(self,entity):
        label = self.query_freebase_label(entity)
        return [label]
    
    def find_relation(self,entity,query,limit=10):
        results = self.query_relation(entity)
        label_to_path = {}
        for path in results:
            label = self.get_relation_label(path)
            label_to_path[label] = path
        labels = list(label_to_path.keys())
        data_json = json.dumps({"question":query,"labels":labels})
        headers = {
            "Content-Type": "application/json; charset=UTF-8",
            }
        result = requests.post("http://210.28.134.34:6220/score_set",data_json,headers=headers)
        scores = result.json()["scores"]
        paths_with_score = []
        for i in range(len(labels)):
            path = label_to_path[labels[i]]
            score = float(scores[i])
            paths_with_score.append([path,score])
        paths_with_score.sort(key = lambda x:x[1],reverse=True)
        paths_with_score = paths_with_score[:limit]
        return paths_with_score

    def query_relation(self,entity):
        entity = self.to_global_name(entity)
        sparql_str = """
        select distinct ?p1
        where{
        %s ?p1 ?x
        filter(?p1 != rdf:type && ?p1 != rdfs:label)
        filter(?p1 != fb:type.object.type && ?p1 != fb:type.object.instance)
        filter( !regex(?p1,\"wikipedia\",\"i\"))
        filter( !regex(?p1,\"type.object\",\"i\"))
        filter( !regex(?p1,\"common.topic.\",\"i\"))
        filter( !regex(?p1,\"_id\",\"i\"))
        filter( !regex(?p1,\"#type\",\"i\"))
        filter( !regex(?p1,\"#label\",\"i\"))
        filter( !regex(?p1,\"_id\",\"i\"))
        filter( !regex(?p1,\"/ns/freebase.\",\"i\"))
        filter( regex(?p1,\"^http://rdf.freebase.com/ns/\",\"i\"))
        }
        """ % (entity)
        sparql_str = self.add_prefix(sparql_str)
        result = self.run_select(sparql_str)
        query_result = []
        for row in result["results"]["bindings"]:
            prop = row["p1"]["value"]
            query_result.append(prop)
        return query_result
    
    def query_object(self,entity,relation):
        entity = self.to_global_name(entity)
        relation = self.to_global_name(relation)
        query_result = []
        sparql_str = """
        select distinct ?object ?label
        where{
            %s %s ?object.
            optional {
                ?object rdfs:label ?label.
                filter langMatches(lang(?label),"en")
            }
        }
        """ % (entity,relation)
        sparql_str = self.add_prefix(sparql_str)
        result = self.run_select(sparql_str)
        for row in result["results"]["bindings"]:
            row_result = []
            if "object" in row.keys():
                row_result.append(row["object"]["value"])
            if "label" in row.keys():
                row_result.append(row["label"]["value"])
            query_result.append(row_result)
        return query_result

    def query_onehop_by_er(self,entity,relation):
        entity = self.to_global_name(entity)
        relation = self.to_global_name(relation)
        sparql_str = """
        select distinct ?mid_ent ?prop ?object ?label
        where {
            %s %s ?mid_ent.
            ?mid_ent ?prop ?object.
            filter(?prop != rdf:type && ?prop != rdfs:label)
            filter(?prop != fb:type.object.type && ?prop != fb:type.object.instance)
            filter( !regex(?prop,\"wikipedia\",\"i\"))
            filter( !regex(?prop,\"type.object\",\"i\"))
            filter( !regex(?prop,\"common.topic.\",\"i\"))
            filter( !regex(?prop,\"_id\",\"i\"))
            filter( !regex(?prop,\"#type\",\"i\"))
            filter( !regex(?prop,\"#label\",\"i\"))
            filter( !regex(?prop,\"_id\",\"i\"))
            filter( !regex(?prop,\"/ns/freebase.\",\"i\"))
            filter( regex(?prop,\"^http://rdf.freebase.com/ns/\",\"i\"))
            optional {
                ?object rdfs:label ?label.
                filter langMatches(lang(?label),"en")
            }
        }
        """ % (entity,relation)
        sparql_str = self.add_prefix(sparql_str)
        result = self.run_select(sparql_str)
        mid_pv = {}
        for row in result["results"]["bindings"]:
            mid = row["mid_ent"]["value"]
            pv_record = None
            if mid in mid_pv.keys():
                pv_record = mid_pv[mid]
            else:
                pv_record = []
                mid_pv[mid] = pv_record
            if "label" in row.keys():
                pv_record.append([row["prop"]["value"],row["object"]["value"],row["label"]["value"]])
            else:
                pv_record.append([row["prop"]["value"],row["object"]["value"],None])
        return mid_pv.values()
    
    def query_relation_by_er(self,entity,relation):
        entity = self.to_global_name(entity)
        relation = self.to_global_name(relation)
        query_result = []
        sparql_str = """
        select distinct ?prop
        where {
            %s %s ?mid_ent.
            ?mid_ent ?prop ?object.
            filter(?prop != rdf:type && ?prop != rdfs:label)
            filter(?prop != fb:type.object.type && ?prop != fb:type.object.instance)
            filter( !regex(?prop,\"wikipedia\",\"i\"))
            filter( !regex(?prop,\"type.object\",\"i\"))
            filter( !regex(?prop,\"common.topic.\",\"i\"))
            filter( !regex(?prop,\"_id\",\"i\"))
            filter( !regex(?prop,\"#type\",\"i\"))
            filter( !regex(?prop,\"#label\",\"i\"))
            filter( !regex(?prop,\"_id\",\"i\"))
            filter( !regex(?prop,\"/ns/freebase.\",\"i\"))
            filter( regex(?prop,\"^http://rdf.freebase.com/ns/\",\"i\"))
        }
        """ % (entity,relation)
        sparql_str = self.add_prefix(sparql_str)
        result = self.run_select(sparql_str)
        for row in result["results"]["bindings"]:
            query_result.append([row["prop"]["value"]])
        return query_result

    def run_select(self,query):
        self.sparql.setQuery(query)
        result = self.sparql.query().convert()
        return result
    
    def to_global_name(self,mid):
        if mid.startswith("http"):
            return "<" + mid + ">"
        elif mid.startswith("fb:"):
            return mid
        return "fb:" + mid

    def add_prefix(self,query):
        prefix = '''PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
                PREFIX foaf: <http://xmlns.com/foaf/0.1/>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
                PREFIX owl: <http://www.w3.org/2002/07/owl#>
                PREFIX fb: <http://rdf.freebase.com/ns/>\n'''
        return prefix + query
    
    def get_relation_label(self,relation):
        result = relation.split("/")[-1].split(".")[-1]
        result = result.replace("_"," ")
        label = " ".join(result.split())
        return label
    
    def query_freebase_label(self,entity):
        entity = self.to_global_name(entity)
        sparql_str = """
        select distinct ?label
        where{
        %s rdfs:label ?label
        filter(langMatches(lang(?label),"en"))
        }
        """ % (entity)
        sparql_str = self.add_prefix(sparql_str)
        result = self.run_select(sparql_str)
        return result["results"]["bindings"][0]["label"]["value"]

if __name__ == '__main__':
    sparql_util = SparqlUtil()
    print(sparql_util.find_entity('William O. Schaefer Elementary School'))
    # print(sparql_util.query_relation_by_query('m.0dl567',"birth date"))