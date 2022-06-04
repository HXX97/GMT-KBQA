"""
Modified on the basis of:
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import json


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


def tokenize_s_expr(expr):
    expr = expr.replace('(', ' ( ')
    expr = expr.replace(')', ' ) ')
    toks = expr.split()
    toks = [x for x in toks if len(x)]
    return toks


class ASTNode:
    def __init__(self, construction, val, fields):
        self.construction = construction
        self.val = val
        self.fields = fields
        # determined after construction
        self.depth = -1
        self.level = -1
    
    def assign_depth_and_level(self, level=0):
        self.level = level
        if self.fields:
            max_depth = max([x.assign_depth_and_level(level + 1) for x in self.fields])
            self.depth = max_depth + 1
        else:
            self.depth = 0
        return self.depth
    
    @classmethod
    def build(cls, tok, fields, atom_type=None):
        """
        只有 build 原子元素时，需要考虑 atom_type
        """
        if tok == 'AND':
            return AndNode(fields)
        elif tok == 'R':
            return RNode(fields)
        elif tok == 'COUNT':
            return CountNode(fields)
        elif tok == 'JOIN':
            return JoinNode(fields)
        elif tok == 'TC':
            return TimeNode(fields)
        elif tok in ['le', 'lt', 'ge', 'gt']:
            return CompNode(tok, fields)
        elif tok in ['ARGMIN', 'ARGMAX']:
            return ArgNode(tok, fields)
        elif tok in ['[ENT]', '[REL]', '[LIT]']:
            if atom_type is not None and tok != atom_type:
                return None
            return AtomNode(tok, fields)
    
    def logical_form(self):
        if self.depth == 0:
            return self.val
        else:
            fields_str = [x.logical_form() for x in self.fields]
            return ' '.join(['(', self.val] + fields_str +  [')'])
    
    # nothing special. just fit legacy code input syle
    def compact_logical_form(self):
        lf = self.logical_form()
        return lf.replace('( ', '(').replace(' )', ')')
    
    def skeleton_form(self):
        if self.depth == 0:
            return self.construction
        else:
            fields_str = [x.skeleton_form() for x in self.fields]
            return ' '.join(['(', self.construction] + fields_str +  [')'])
    
    def __str__(self):
        return self.logical_form()

    def __repr__(self):
        return self.logical_form()

class AndNode(ASTNode):
    def __init__(self, fields):
        super().__init__('AND', 'AND', fields)

class RNode(ASTNode):
    def __init__(self, fields):
        super().__init__('R', 'R', fields)

class CountNode(ASTNode):
    def __init__(self, fields):
        super().__init__('COUNT', 'COUNT', fields)

class JoinNode(ASTNode):
    def __init__(self, fields):
        super().__init__('JOIN', 'JOIN', fields)

class ArgNode(ASTNode):
    def __init__(self, val, fields):
        super().__init__('ARG', val, fields)

class CompNode(ASTNode):
    """Compare: GT, GE, LT, LE"""
    def __init__(self, val, fields):
        super().__init__('COMP', val, fields)

class TimeNode(ASTNode):
    """Time Constraint"""
    def __init__(self, fields):
        super().__init__('TC', 'TC', fields)

class AtomNode(ASTNode):
    """
    Three kind of Atoms:
    [ENT], [REL], [LIT]
    """
    def __init__(self, val, fields):
        super().__init__('ATOM', val, fields)

def _consume_a_node(tokens, cursor, atom_type=None):
    is_root = cursor == 0
    if cursor >= len(tokens):
        return None, cursor
    cur_tok = tokens[cursor]
    cursor += 1
    if cur_tok == '(':
        node, cursor = _consume_a_node(tokens, cursor)
        if node is None or cursor >= len(tokens) or tokens[cursor] != ')':
            return None, cursor
        cursor += 1
    elif cur_tok == 'AND':
        # left, right, all unary
        left, cursor = _consume_a_node(tokens, cursor)
        if left is None:
            return None, cursor
        right, cursor = _consume_a_node(tokens, cursor)
        if right is None:
            return None, cursor
        node = ASTNode.build(cur_tok, [left, right])
    elif cur_tok == 'JOIN':
        # if cur is unary, right unary, else right binary
        left, cursor = _consume_a_node(tokens, cursor, atom_type='[REL]')
        if left is None:
            return None, cursor
        right, cursor = _consume_a_node(tokens, cursor)
        if right is None:
            return None, cursor
        node = ASTNode.build(cur_tok, [left, right])
    elif cur_tok == 'ARGMIN' or cur_tok == 'ARGMAX':
        left, cursor = _consume_a_node(tokens, cursor)
        if left is None:
            return None, cursor
        right, cursor = _consume_a_node(tokens, cursor, atom_type='[REL]')
        if right is None:
            return None, cursor
        node = ASTNode.build(cur_tok, [left, right])
    elif cur_tok == 'le' or cur_tok == 'lt' or cur_tok == 'ge' or cur_tok == 'gt':
        left, cursor = _consume_a_node(tokens, cursor)
        if left is None:
            return None, cursor
        right, cursor = _consume_a_node(tokens, cursor, atom_type='[LIT]')
        if right is None:
            return None, cursor
        node = ASTNode.build(cur_tok, [left, right])
    elif cur_tok == 'TC':
        left, cursor = _consume_a_node(tokens, cursor)
        if left is None:
            return None, cursor
        relation, cursor = _consume_a_node(tokens, cursor, atom_type='[REL]')
        if relation is None:
            return None, cursor
        time_point, cursor = _consume_a_node(tokens, cursor, atom_type='[LIT]')
        if time_point is None:
            return None, cursor
        node = ASTNode.build(cur_tok, [left, relation, time_point])
    elif cur_tok == 'R':
        child, cursor = _consume_a_node(tokens, cursor, atom_type='[REL]')
        if child is None:
            return None, cursor
        node =  ASTNode.build(cur_tok, [child])
    elif cur_tok == 'COUNT':
        child, cursor = _consume_a_node(tokens, cursor)
        if child is None:
            return None, cursor
        node =  ASTNode.build(cur_tok, [child])
    else:
        # Atom Node
        node =  ASTNode.build(cur_tok, [], atom_type)
    if is_root:
        if node is None:
            return None, cursor
        node.assign_depth_and_level()

    return node, cursor

# top lvel: and, arg, count, cant be JOIN, LE, R
def parse_s_expr(expr):
    # preprocess
    expr = expr.replace('LESS THAN', 'lt').replace('LESS EQUAL', 'le').replace('GREATER THAN', 'gt').replace('GREATER EQUAL', 'ge')
    expr = expr.replace('[ENT]', ' [ENT] ').replace('[REL]', ' [REL] ').replace('[LIT]', ' [LIT] ')
    tokens = tokenize_s_expr(expr)
    ast, _ = _consume_a_node(tokens, 0)
    # assert ' '.join(tokens) == ast.logical_form()
    return ast  


def test_coverage(file_path):
    """数据集中的 golden normed masked sexpr, 是否都能通过检查""" 
    dataset = load_json(file_path)
    normed_masked_sexprs = list(set([data["normed_all_masked_sexpr"] for data in dataset if data["normed_all_masked_sexpr"]!='null']))
    missed_sexprs = []
    for normed_sexpr in normed_masked_sexprs:
        ast = parse_s_expr(normed_sexpr)
        if ast is None:
            missed_sexprs.append(normed_sexpr)
    print(missed_sexprs, len(missed_sexprs))


def unit_test():
    for split in ['train', 'dev', 'test']:
        print(split, 'CWQ')
        CWQ_path = '/home3/xwu/workspace/QDT2SExpr/CWQ/data/CWQ/final/merged/CWQ_{}.json'.format(split)
        test_coverage(CWQ_path)
    for split in ['train', 'test']:
        print(split, 'WebQSP')
        WebQSP_path = '/home3/xwu/workspace/QDT2SExpr/CWQ/data/WebQSP/final/merged/WebQSP_{}.json'.format(split)
        test_coverage(WebQSP_path)


def atom_test():
    test_cases = [
        '( JOIN ( R[ENT] ) ( JOIN ( R[REL] ) ( JOIN[REL] ( JOIN[REL][ENT] ) ) ) )',
        '( JOIN ( R[LIT] ) ( JOIN ( R[REL] ) ( JOIN[REL] ( JOIN[REL][ENT] ) ) ) )',
        '( AND ( JOIN[REL][ENT] ) ( JOIN ( R[REL] ) ( JOIN[ENT][ENT] ) ) )',
        '( JOIN ( R[REL] ) ( JOIN[LIT] ( JOIN[REL][ENT] ) ) )',
        '( AND ( lt ( JOIN[REL][REL] )[REL] ) ( JOIN ( R[REL] ) ( TC ( AND ( JOIN[REL][ENT] ) ( JOIN ( R[REL] )[ENT] ) )[REL][LIT] ) ) )',
        '( AND ( lt ( JOIN[REL][REL] )[LIT] ) ( JOIN ( R[REL] ) ( TC ( AND ( JOIN[REL][ENT] ) ( JOIN ( R[REL] )[ENT] ) )[REL][ENT] ) ) )',
        '( AND ( lt ( JOIN[REL][REL] )[LIT] ) ( JOIN ( R[REL] ) ( TC ( AND ( JOIN[REL][ENT] ) ( JOIN ( R[REL] )[ENT] ) )[REL][REL] ) ) )'
    ]

    for case in test_cases:
        ast = parse_s_expr(case)
        assert ast is None, print(case, ast)

if __name__=='__main__':
    # s_expr = "( JOIN ( R [REL] ) ( TC ( AND ( JOIN [REL] [ENT] ) ( JOIN ( R [REL] ) [ENT] ) ) [REL] [LIT] ) )"
    # ast = parse_s_expr(s_expr)
    # print('sexpr: {}'.format(s_expr))
    # if ast is None:
    #     print('syntax error')
    # else:
    #     skeleton = ast.skeleton_form()
    #     print('skeleton: {}'.format(skeleton))
    #     logical_form = ast.logical_form()
    #     print('logical_form: {}'.format(logical_form))
    # for split in ['train', 'test']:
    #     test_coverage(split)
    
    unit_test()
    # atom_test()

