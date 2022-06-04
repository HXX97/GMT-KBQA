"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import re


def tokenize_s_expr(expr):
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    return toks

def extract_mentioned_entities(expr):
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split(' ')
    toks = [x for x in toks if len(x)]
    entitiy_tokens = []
    for t in toks:
        # normalize entity
        if t.startswith('m.') or t.startswith('g.'):
            entitiy_tokens.append(t)
    return entitiy_tokens

def extract_mentioned_entities_from_sparql(sparql):
    """extract entity from sparql"""
    sparql = sparql.replace('(',' ( ').replace(')',' ) ')
    toks = sparql.split(' ')
    toks = [x for x in toks if len(x)]
    entity_tokens = []
    for t in toks:
        if t.startswith('ns:m.') or t.startswith('ns:g.'):
            entity_tokens.append(t[3:])
        
    entity_tokens = list(set(entity_tokens))
    return entity_tokens

def extract_mentioned_relations_from_sparql(sparql):
    """extract relation from sparql"""
    sparql = sparql.replace('(',' ( ').replace(')',' ) ')
    toks = sparql.split(' ')
    toks = [x for x in toks if len(x)]
    relation_tokens = []
    for t in toks:
        if (re.match("ns:[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-Z_]*",t.strip()) 
            or re.match("ns:[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-Z_]*",t.strip())):
            relation_tokens.append(t[3:])
    
    relation_tokens = list(set(relation_tokens))
    return relation_tokens


def extract_mentioned_relations_from_sexpr(sexpr):
    sexpr = sexpr.replace('(',' ( ').replace(')',' ) ')
    toks = sexpr.split(' ')
    toks = [x for x in toks if len(x)]
    relation_tokens = []

    for t in toks:
        if (re.match("[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-z_]*",t.strip()) 
            or re.match("[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-Z_]*\.[a-zA-Z_]*",t.strip())):
            relation_tokens.append(t)
    relation_tokens = list(set(relation_tokens))
    return relation_tokens


if __name__=="__main__":
    # sparql = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.04yd0fh)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.04yd0fh ns:film.actor.film ?y .\n?y ns:film.performance.film ?x .\n?x ns:film.film.starring ?c .\n?c ns:film.performance.character ns:g.125_b_w7k . \n}"
    sparql = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ?c)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\n?c ns:sports.sports_team.championships ns:m.04tfqf . \n?c ns:sports.sports_team.championships ?x .\n?x ns:time.event.end_date ?sk0 .\n}\nORDER BY xsd:datetime(?sk0)\nLIMIT 1\n"
    print(extract_mentioned_entities_from_sparql(sparql=sparql))
    print(extract_mentioned_relations_from_sparql(sparql=sparql))

    sexpr = '(JOIN (R government.government_position_held.office_holder) (TC (AND (JOIN government.government_position_held.office_position_or_title m.0j5wjnc) (JOIN (R government.governmental_jurisdiction.governing_officials) (JOIN location.country.national_anthem (JOIN government.national_anthem_of_a_country.anthem m.0gg95zf)))) government.government_position_held.from NOW))'
    print(extract_mentioned_relations_from_sexpr(sexpr))

    