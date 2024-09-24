import json
import os
import numpy as np
import random
import argparse
import shutil
from itertools import chain
from pysat.solvers import Solver, Minisat22


def solve_sat(cnf):
    """Solve a SAT instance in CNF form"""
    with Solver('m22', bootstrap_with=cnf.clauses) as g:
        return bool(g.solve()), g.get_model()
    

def ilit_to_var_sign(x):
    assert (abs(x) > 0)
    var = abs(x) - 1
    sign = x < 0
    return var, sign


def ilit_to_vlit(x, n_vars):
    assert (x != 0)
    var, sign = ilit_to_var_sign(x)
    if sign:
        return var + n_vars
    else:
        return var


def generate_k_iclause(n, k):
    vs = np.random.choice(n, size=min(n, k), replace=False)
    return [int(v + 1) if random.random() < 0.5 else int(-(v + 1)) for v in vs]


def gen_iclause_pair(n, p_k_2, p_geo):
    solver = Minisat22()
    model = -1
    iclauses = []
    while True:
        k_base = 1 if random.random() < p_k_2 else 2
        k = k_base + np.random.geometric(p_geo)
        iclause = generate_k_iclause(n, k)  # Generate clause with k variables
        solver.add_clause(iclause)
        is_sat = solver.solve()
        if is_sat:
            model = solver.get_model()
            iclauses.append(iclause)  # Keep adding clauses until is unsat
        else:
            break
    iclause_unsat = iclause  # Save the unsat clause
    iclause_sat = [- iclause_unsat[0]] + iclause_unsat[1:]  # Negate one literal of the unsat clause to make it sat
    return iclauses, iclause_unsat, iclause_sat, model  # Return the sat cnf, with the two additional unsat and sat clause


def to_dimacs( n, clause_list):
        m = len(clause_list)
        body = ''
        for clause in clause_list:
            body += (str(clause)[1:-1].replace(',', '')) + ' 0\n'
        return 'p cnf ' + str(n) + ' ' + str(m) + '\n' + body


def main(args):

    random.seed(args.seed)
    np.random.seed(args.seed)

    n_cnt = args.max_n - args.min_n + 1  # different number of n
    problems_per_n = int(args.n_pairs * 1.0 / n_cnt)  # number of instances per each n
    list_n_vars = list(chain.from_iterable((n_vars,)*problems_per_n for n_vars in range(args.min_n, args.max_n+1)))
    random.shuffle(list_n_vars)

    name = 'SR-' + str(args.min_n) + '-' + str(args.max_n)

    out_dir = '/nfs/students/aor/datasets/' + name
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    print(f"Generating {name} | sat_only={args.sat_only}")
    
    for p_idx, n_vars in enumerate(list_n_vars):
        print('Processing Problem ', p_idx)

        iclauses, iclause_unsat, iclause_sat, solution = gen_iclause_pair(n_vars, args.p_k_2, args.p_geo)
        
        dimacs_sat_file = f"dimacs_{p_idx}_sat=1.DIMACS"
        sat_cnf = iclauses + [iclause_sat]
        with open(os.path.join(out_dir, dimacs_sat_file), 'w') as g:
            g.write(to_dimacs(n_vars, sat_cnf) + '\n')

        sol_file_name = f"{dimacs_sat_file.split('sat=')[0]}sol.json"
        with open(os.path.join(out_dir, sol_file_name), 'w') as g:
            json.dump(solution, g)

        if not args.sat_only:
            dimacs_unsat_file = f"dimacs_{p_idx}_sat=0.DIMACS"
            unsat_cnf = iclauses + [iclause_unsat]
            with open(os.path.join(out_dir, dimacs_unsat_file), 'w') as g:
                g.write(to_dimacs(n_vars, unsat_cnf) + '\n')
    
    print("Finished succesfully!")
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate SR(U(start, end)) data')
    parser.add_argument('-n_pairs', default=10000, help='How many problem pairs to generate', type=int)
    parser.add_argument('-min_n', default=3, help='start value for number of variables', type=int)
    parser.add_argument('-max_n', default=10, help='end value for number of variables', type=int)
    parser.add_argument('-p_k_2', default=0.3, type=float)
    parser.add_argument('-p_geo', default=0.4, type=float)
    parser.add_argument('--sat_only', default=False, action='store_true')
    parser.add_argument('-seed', default=0, type=int)
    parser.add_argument('-remove_ss', default=False, type=bool)
    args = parser.parse_args()

    main(args)