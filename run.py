from utils import write_solution
from ga import BeamSolver
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', default='./data')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--strategy', choices=['simple', 'astar', 'beam'], required=True)
    args = parser.parse_args()

    if args.strategy == 'simple':
        order_keys = ['server']
    elif args.strategy == 'astar':
        # {'server', 'max_serve', 'max_bands', 'next_step_max_bands', '95_free_lunch', 'estimate_lookahead'}
        order_keys = ['95_free_lunch', 'estimate_lookahead', 'max_serve', 'max_bands']
    else:
        raise NotImplementedError()

    solver = BeamSolver(args.data, 1, order_keys, seed=args.seed)
    solver.calculate(0, debug=True)