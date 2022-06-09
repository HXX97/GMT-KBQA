"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


import pickle
import json
import os
import shutil
import re


def dump_to_bin(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)


def load_bin(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


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


def mkdir_f(prefix):
    if os.path.exists(prefix):
        shutil.rmtree(prefix)
    os.makedirs(prefix)


def mkdir_p(prefix):
    if not os.path.exists(prefix):
        os.makedirs(prefix)


illegal_xml_re = re.compile(u'[\x00-\x08\x0b-\x1f\x7f-\x84\x86-\x9f\ud800-\udfff\ufdd0-\ufddf\ufffe-\uffff]')
def clean_str(s: str) -> str:
    """remove illegal unicode characters"""
    return illegal_xml_re.sub('',s)