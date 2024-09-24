import json
import os
import time
import numpy as np
import pysat
from pysat.formula import CNF
from pysat.solvers import Solver
from threading import Timer
from itertools import product, islice
from dpll import dp

def _solve(cnf, solver='glucose4', time_limit=None):
    """
    First implementation of SAT Solver without interruption
    """
    with Solver(solver, bootstrap_with=cnf.clauses, use_timer=True) as g:
        sat = int(bool(g.solve()))
        t = g.time()
        assert sat == 1, "SAT instance not sat!"
    return int(bool(t <= time_limit))


def solve(cnf, solver, time_limit=None):
    """
    Improved version with interruption, to speed-up the running time. 
    Worst case ~ 500*3 = 1500s = 25m
    """
    #print(f"n: {cnf.nv} | m: {len(cnf.clauses)}")
    if solver == 'dp':
        sat = dp(cnf)
    elif solver == 'dpll':
        return 0, 0
        # sat = dpll(cnf)
    else:
        with Solver(solver, bootstrap_with=cnf.clauses, use_timer=True) as g:
            timer = Timer(time_limit, lambda x: x.interrupt(), [g])
            timer.start()
            sat = g.solve_limited(expect_interrupt=True)
            g.clear_interrupt()
    solved = int(sat != None) # We are interested in % of solved instances either sat or unsat within the time limit
    # assert not solved or sat == 1, "SAT instance not sat!" # We are sure the instances are all sat     
    return (solved, sat)


def solve_all(cnfs, solver='glucose4'):
    """
    Solve all instances, cnf formulas with solver.
    @param solver: ['glucose3', 'glucose4', 'minisat22', ...]
    """
    solved, sat = zip(*list(solve(cnf, solver, time_limit=3) for cnf in cnfs))
    return sum(solved), sum(sat)



def main():
    debug = True
    idxs = 1 if debug else ...
    solvers = ['glucose4', 'minisat22', 'dp', 'dpll']
    solvers_acc = {k:{} for k in solvers}
    for i, alpha in enumerate(np.arange(7, 10.5, 0.5)[:1]):
        print("\n", f"4-SAT-100 alpha {alpha}:", "\n---------------------")
        dataset = f"4-SAT-100_{i}_{alpha}_{alpha+0.5}"
        #directory_path = f"../../PDP-Solver/datasets/test/{dataset}"
        directory_path = "/nfs/students/aor/SR-3-10"
        cnfs = list(islice((CNF(from_file=os.path.join(directory_path, file)) for file in os.listdir(directory_path) if file.endswith(".DIMACS")), 200))
        print("n_problems:", len(cnfs))
        for solver in solvers:
            st = time.time()
            solved, sat = solve_all(cnfs, solver=solver)
            rt = time.time() - st
            print("solver:", solver, "| solved:", solved, "| sat:", sat, "| time:", rt, "s")
            solvers_acc[solver][alpha] = solved
    
    with open("result_sat_solvers", 'w') as f:
        json.dump(solvers_acc, f)


if __name__ == "__main__":
    main()
