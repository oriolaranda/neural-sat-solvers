import numpy as np
from .dp import compute_ordering
from typing import Tuple, List, Dict, Set, Union


# Custom Types
Literal = int
Clause = Set[Literal]
Solution = Set[Literal]

SAT = True
UNSAT = False


def complete_solution(part_sol: Solution, n: int) -> Solution:
    # complete the solution with the variables that can be (0, 1)
    sorted_sol = np.array(sorted(list(part_sol), key=lambda x: abs(x)))
    complete_sol = np.arange(1, n+1)
    complete_sol[np.abs(sorted_sol)-1] = sorted_sol
    return complete_sol.tolist()


def conditioned_v(clauses: List[Clause], l: Literal) -> List[Clause]:
    conditioned = [c - {-l} for c in clauses if l not in c]
    return conditioned


def dpll_(cnf, ordering_type=2):
    """Wrapper"""
    
    def _dpll_(clauses: List[Clause], d: int) -> Union[bool, Solution]:
        """
        DPLL- recursive implementation with clauses and depth d.
        Use list of sets because multisets are not available   
        """
        # If there are no more clauses, end recursion
        if clauses == []:
            return set({})
        # If we can deduce the empty clause return UNSAT
        if set({}) in clauses:
            return UNSAT
        
        # Select current depth level decision variable
        v = ordering[d]
        
        # Condition true
        sol = _dpll_(conditioned_v(clauses, v), d+1)
        if sol != UNSAT:
            return sol | {v}
        
        # Condition False
        sol = _dpll_(conditioned_v(clauses, -v), d+1)
        if sol != UNSAT:
            return sol | {-v}
        
        return UNSAT
    
    ordering, _ = compute_ordering(cnf, ordering_type)
    clauses = [set(c) for c in cnf.clauses]
    result = _dpll_(clauses, 0)
    sat = bool(result)
    sol = -1 if not sat else complete_solution(result, cnf.nv)
    return sat, sol


def iter_dpll_(cnf, ordering_type=2):
    """Wrapper"""
    
    def _iter_dpll_(clauses: List[Clause]) -> Union[bool, Solution]:
        """
        DPLL- iterative implementation. Use list of sets because 
        multisets are not available.
        """
        stack = [(clauses, 0, set({}))]

        while stack:
            clauses, d, sol = stack.pop()

            if clauses == []:
                return sol

            if set({}) in clauses:
                continue

            v = ordering[d]
            stack.append((conditioned_v(clauses, v), d+1, sol | {v}))
            stack.append((conditioned_v(clauses, -v), d+1, sol | {-v}))

        return UNSAT
    
    ordering, _ = compute_ordering(cnf, ordering_type)
    clauses = [set(c) for c in cnf.clauses]
    result = _iter_dpll_(clauses, 0)
    sat = bool(result)
    sol = -1 if not sat else complete_solution(result, cnf.nv)
    return sat, sol









def dpll(cnf):
    "DPLL implementation with unit-resolution"
    i, f = unit_resolution(cnf)
    if f == []:
        return i
    if [] in f:
        return False
    v = choose_literal(f)
    l = dpll(conditioned(f, v))
    if l != False:
        return l | i | {v}
    l = dpll(conditioned(f, -v))
    if l != False:
        return l | i | {-v}
    return False