import json
import os
import time
import numpy as np
from tqdm import tqdm
from pysat.formula import CNF
from pysat.solvers import Solver
from threading import Timer
from itertools import product
import torch
from typing import Tuple, List, Dict, Union

SAT = True
UNSAT = False

def compute_ordering(cnf, opt=0):
    variables = range(1, cnf.nv+1)
    if opt == 0: # Natural ordering
        scores = {v: v for v in variables}
    elif opt == 1: # Random ordering
        scores = {k: v for v, k in enumerate(np.random.permutation(variables))}
    elif opt == 2: # Count occurrances of variables
        scores = count_occurrances(cnf)
    else:
        raise ValueError("Option not implemented!")
    
    ordering = sorted(variables, key=lambda x: scores[abs(x)])
    fixed_ord = {k:v for v, k in enumerate(ordering)}
    cmp_func = lambda x: fixed_ord[abs(x)]
    return ordering, cmp_func

def count_occurrances(cnf) -> Dict[int, int]:
    counter = {k: 0 for k in range(1, cnf.nv+1)}
    for clause in cnf.clauses:
        for literal in clause:
            counter[abs(literal)] -= 1 #count in negative to sort
    return counter

def is_empty(clause) -> bool:
    return clause == tuple([])

def is_trivial(c) -> bool:
    return any(not(v1 + v2) for v1, v2 in zip(c, c[1:]))

def merge(c1, c2, cmp_func) -> Tuple[int]:
    return tuple(sorted(list(set(c1+c2)), key=cmp_func))

def get_resolvent_clauses(bucket, v, cmp_func):
    pos = (c[1:] for c in bucket[v] if c[0] >= 0)
    neg = (c[1:] for c in bucket[v] if c[0] < 0)
    resolv_clauses = {merge(c1, c2, cmp_func) for c1, c2 in product(pos, neg)}
    resolv_clauses = (x for x in resolv_clauses if not is_trivial(x))
    return resolv_clauses

def dp(cnf, ordering_type=2) -> bool:
    "DP also known by DR (Direct Resolution). Implementation with bucket elimination"
    # Choose an ordering
    ordering, cmp_func = compute_ordering(cnf, ordering_type)
    
    # Initialize empty buckets for each variable
    bucket = {v: set() for v in ordering}

    # Put each clause to the corresponding buckets following the 
    # variable ordering
    for c in cnf.clauses:
        c_sorted = sorted(c, key=cmp_func)
        v = abs(c_sorted[0])
        bucket[v] = bucket[v] | {tuple(c_sorted)}
    
    # Apply resolution to the buckets in order
    for v in ordering:
        if bucket[v]:
            v_resolvent_clauses = get_resolvent_clauses(bucket, v, cmp_func)
            for c in v_resolvent_clauses:
                if is_empty(c):
                    return UNSAT # The formula cannot imply empty clause
                u = abs(c[0])
                bucket[u] = bucket[u] | {c}
    return SAT