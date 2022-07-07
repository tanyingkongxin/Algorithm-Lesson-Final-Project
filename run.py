from utils import write_solution, tools_get_time
from solver import Solver
from argparse import ArgumentParser
import time

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', default='./data')
    parser.add_argument('--seed', default=0)
    parser.add_argument('--strategy', choices=['simple', 'astar'], required=True)
    parser.add_argument('--lift', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=0.5)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    if args.strategy == 'simple':
        order_keys = ['server']
    elif args.strategy == 'astar':
        # {'server', 'max_serve', 'max_bands', 'next_step_max_bands', '95_free_lunch', 'estimate_lookahead'}
        order_keys = ['95_free_lunch', 'max_serve', 'estimate_lookahead', 'max_bands']
    else:
        raise NotImplementedError()

    solver = Solver(args.data, args.lift, args.epsilon, order_keys, seed=args.seed)
    if args.output is None:
        args.output = f"{tools_get_time()}.psize{solver.max_time_num}.solution.txt"
    print(f'{tools_get_time()} starting searching')
    start_time = time.time()
    iter_servers, iter_clients = solver.calculate(0, debug=False)
    end_time = time.time() - start_time
    write_solution(iter_clients, args.output, solver.max_time_num)
    print(f'{tools_get_time()} done cost {end_time} seconds! Results saved to {args.output}')