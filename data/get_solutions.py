import os
from os import path
import numpy as np
import random
import argparse
import shutil
import json
from itertools import chain
from pysat.solvers import Solver
from pysat.formula import CNF


def get_filter_func(sat_only):
    """Filter Dimacs files optionally just SAT instances"""
    def _filter(file):
        sat = 'sat=1' if sat_only else ''
        return file.endswith(".DIMACS") and (sat in file)
    return _filter

def solve_sat(cnf):
    """Solve a SAT instance in CNF form"""
    with Solver('m22', bootstrap_with=cnf.clauses) as g:
        return bool(g.solve()), g.get_model()


def main(args):
    path = args.path
    filter_func = get_filter_func(sat_only=True)
    files = list(file for file in os.listdir(path) if filter_func(file))

    for i, file in enumerate(files):
        print(f"Processing {i} {file}")
        cnf = CNF(from_file=path+file)
        sat, solution = solve_sat(cnf)
        assert sat, "Formula UNSAT!"
        new_file = f"{file.split('sat=')[0]}sol.json"
        with open(path + new_file, 'w') as f:
            json.dump(solution, f)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Solve satisfiable DIMACS instances and save the solution')
    parser.add_argument('--path', default=None, 
                        help='Path to DIMACS folder')
    args = parser.parse_args()

    main(args)

