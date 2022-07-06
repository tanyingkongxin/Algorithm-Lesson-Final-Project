import random
import math
import copy
import numpy as np
from tqdm import tqdm
from utils import softmax, read_data, write_solution


class Client():
    def __init__(self, name: str, demands: [int]):
        self.name = name
        self.demands = demands
        self.__original_demands = copy.deepcopy(demands)
        self.history = [{} for _ in self.demands]

    def update(self, server_name, time_t, bands):
        assert bands <= self.demands[time_t]
        self.demands[time_t] -= bands
        if server_name not in self.history[time_t]:
            self.history[time_t][server_name] = 0
        self.history[time_t][server_name] += bands

    def reset(self, time_t):
        self.demands[time_t] = self.__original_demands[time_t]
        self.history[time_t] = {}

    def __repr__(self):
        return self.name

class Server():
    def __init__(self, name: str, bandwidth: int, qos_constraint, qos_table, max_time_num):
        self.name = name
        self.bandwidth = bandwidth
        self.qos_constraint = qos_constraint
        self.qos_table = qos_table
        self.history_bands = None
        self.reserve_bands = [self.bandwidth for _ in range(max_time_num)]
        self.history_bands = [{} for _ in range(max_time_num)]
        self.max_time_num = max_time_num

    def __repr__(self):
        return f"server_{self.name}"

    def __hash__(self):
        return hash(self.name)

    def reset(self, time_t, client_name):
        self.history_bands[time_t] = {k: v for k, v in self.history_bands[time_t].items() if k != client_name}
        self.reserve_bands[time_t] = self.bandwidth - sum(self.history_bands[time_t].values())

    def afford(self, c: Client, time_t):
        # if time_t in [55]:
        #     print(self.name, self.qos_table[self.name][c.name], self.qos_constraint, self.reserve_bands[time_t])
        return self.qos_table[self.name][c.name] < self.qos_constraint and self.reserve_bands[time_t] > 0

    def execute(self, c: Client, time_t, specified_band=None):
        if specified_band:
            assert specified_band <= self.reserve_bands[time_t] and specified_band <= c.demands[time_t]
            bands = specified_band
        else:
            bands = min(self.reserve_bands[time_t], c.demands[time_t])
        assert bands > 0

        if c.name not in self.history_bands[time_t]:
            self.history_bands[time_t][c.name] = 0
        self.history_bands[time_t][c.name] += bands
        self.reserve_bands[time_t] -= bands
        c.update(self.name, time_t, bands)

    def get_cost(self, end_time):
        assert end_time < len(self.history_bands)
        costs = [sum(item.values()) for item in self.history_bands[:end_time + 1]]
        return sorted(costs)[math.ceil(len(costs) * 0.95) - 1]

    def estimate_lookahead(self, time_t):
        costs = list(sorted([sum(item.values()) for item in self.history_bands[:time_t + 1]]))
        now_cost = costs[math.ceil(len(costs) * 0.95) - 1]
        lookahead = min(int(time_t / self.max_time_num * time_t), len(self.history_bands) - time_t - 1)
        if time_t < int(0.5 * self.max_time_num): lookahead = int(lookahead * 1.15)
        p  = max(time_t / self.max_time_num - 0.3, 0.0) + 0.2
        estimate_band = p * costs[-1] + (1 - p) * self.bandwidth
        costs += [estimate_band] * lookahead
        next_cost = costs[math.ceil(len(costs) * 0.95) - 1]
        return now_cost - next_cost

    def free_lunch(self, time_t):
        already_num, idx = self.get_position()
        if idx + already_num + 1 < self.max_time_num:
            return 1 # free lunch
        else:
            current = max([sum(item.values()) for item in self.history_bands[:time_t + 1]])
            return -(0.9 * current + 0.1 * self.bandwidth)

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return False

    def get_position(self):
        already_num = 0
        for item in self.history_bands[:self.max_time_num]:
            if len(item) > 0:
                already_num += 1
        return already_num, math.ceil(self.max_time_num * 0.95) - 1

class BeamSolver():
    def __init__(self, data_dir, beam, order_keys, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        self.allowed_keys = {'server', 'max_serve', 'max_bands', 'next_step_max_bands', '95_free_lunch', 'estimate_lookahead'}
        for item in order_keys:
            assert item in self.allowed_keys
        self.data_dir = data_dir
        self.beam = beam
        self.order_keys = order_keys
        self.max_time_num, qos_constraint, customers, servers, server_customers_qos = read_data(self.data_dir)
        self.clients = {k: Client(k, v) for k, v in customers.items()}
        self.servers = {k: Server(k, v, qos_constraint, server_customers_qos, self.max_time_num) for k, v in servers.items()}
        self.distribute_func = distribute

    def calculate(self, time_start=0, debug=False):
        with tqdm(desc='solver', total=self.max_time_num - time_start) as pbar:
            for time_t in range(time_start, self.max_time_num):
                if time_t == time_start:
                    server_order_time_t, clients_time_t, cost = self.sample_server_order(self.clients, self.servers, time_t, 1)[0][time_t]
                    iter_servers = {server.name: server for server in server_order_time_t}
                    iter_clients = clients_time_t
                else:
                    server_order_time_t, clients_time_t, cost = self.sample_server_order(iter_clients, iter_servers, time_t, 1)[0][time_t]
                    iter_servers = {server.name: server for server in server_order_time_t}
                    iter_clients = clients_time_t

                expected_costs = [s.get_cost(s.max_time_num - 1) for s in iter_servers.values()]
                pbar.set_postfix({'time': time_t, 'current_cost': cost, 'expected_cost': sum(expected_costs)})
                pbar.update(1)
                if debug:
                    print(expected_costs)

        return iter_servers, iter_clients

    def greedy_distribute(self, servers_order: [Server], clients: {str: Client}, time_t):
        servers_order, clients = copy.deepcopy(servers_order), copy.deepcopy(clients)
        def find_available_servers(servers: [Server], c: Client, time_t, threshold):
            available = {}
            others = {}
            cnt = 0
            for s in servers:
                s: Server
                if s.afford(c, time_t):
                    if s.free_lunch(s.max_time_num - 1) > threshold:
                        cnt += 1
                        available[s] = 0
                    else:
                        others[s] = 0

            for s in available.keys():
                available[s] = min(s.reserve_bands[time_t], 1000)


            return available, others





        max_fail = 100
        for c in clients.values():
            c: Client
            fail = 0
            while fail < max_fail:
                available, others = find_available_servers(servers_order, c, time_t, 0 - 10 ** fail)
                for s in available.keys():
                    if c.demands[time_t] == 0: break
                    lower = math.ceil(c.demands[time_t] / len(available))
                    upper = math.ceil(c.demands[time_t] / (len(available) + len(others)))
                    sp = random.randint(lower, upper)
                    sp = min(sp, s.reserve_bands[time_t], c.demands[time_t])
                    s.execute(c, time_t, specified_band=sp)

                if c.demands[time_t] > 0:
                    fail += 1
                    c.reset(time_t)
                    for s in available:
                        # 消除已经对c分配过的服务器的影响，并不会影响之前分配的c
                        s.reset(time_t, c.name)
                else:
                    break



            if fail == max_fail:
                #             failed to distribute
                return False, -1, None, None

        cost = sum([s.get_cost(time_t) for s in servers_order])
        return True, cost, servers_order, clients








        # for c in clients.values():
        #     c: Client
        #     if c.demands[time_t] > 0:
        #         for s in servers_order:
        #             s: Server
        #             if s.afford(c, time_t):
        #                 s.execute(c, time_t)
        #             if c.demands[time_t] == 0: break
        #         if c.demands[time_t] > 0:
        #             #             failed to distribute
        #             return False, -1, None, None
        # #     successfully distribute
        # cost = sum([s.get_cost(time_t) for s in servers_order])
        # return True, cost, servers_order, clients

    def sample_server_order(self, clients: {str: Client}, servers: {str: Server}, time_t, num):

        servers_order = self.get_server_order(clients, servers, time_t, self.order_keys)
        p = 100
        weights = []
        for i, _ in enumerate(servers_order):
            weights.append(p)
            if i <= int(len(servers_order) / 2):
                p *= 0.9
            else:
                p *= 0.6
        populations = []

        flag, cost, server_order_time_t, clients_time_t = self.distribute_func(servers_order, clients, time_t)

        if flag:
            populations.append({time_t: [server_order_time_t, clients_time_t, cost]})
        else:
            weights = softmax(weights)
            while len(populations) < num:
                temp = np.random.choice(servers_order, len(servers_order), replace=False, p=weights).tolist()
                flag, cost, server_order_time_t, clients_time_t = self.distribute_func(temp, clients, time_t)
                if flag:
                    # temp already distribute time_t bands
                    populations.append({time_t: [server_order_time_t, clients_time_t, cost]})
                    # print(f"cost in time{time_t} is {cost}")

        return populations

    def get_server_order(self, clients: {str: Client}, servers: {str: Server}, time_t, keys):
        assert isinstance(keys, list) and isinstance(keys[0], str)
        temp = list(clients.values())
        temp = sorted(temp, key=lambda c: c.demands[time_t], reverse=True)
        servers_order = []
        for sname, s in servers.items():
            s: Server
            rbands = 0
            cnt = 0
    #         FFD
            for c in temp:
                c: Client
                if s.afford(c, time_t) and rbands + c.demands[time_t] <= s.bandwidth:
                    rbands += c.demands[time_t]
                    cnt += 1
            servers_order.append(
                {'server': s, 'max_serve': cnt, 'max_bands': s.bandwidth, 'next_step_max_bands': rbands, '95_free_lunch': s.free_lunch(s.max_time_num - 1), 'estimate_lookahead': s.estimate_lookahead(time_t)}
            )

        key = lambda item: [item[k] for k in keys]
        servers_order = list(sorted(servers_order, key=key, reverse=True))
        return [item['server'] for item in servers_order]

def distribute(servers_order: [Server], clients: {str: Client}, time_t):
    servers_order, clients = copy.deepcopy(servers_order), copy.deepcopy(clients)
    for c in clients.values():
        c: Client
        if c.demands[time_t] > 0:
            for s in servers_order:
                s: Server
                if s.afford(c, time_t):
                    s.execute(c, time_t)
                if c.demands[time_t] == 0: break
            if c.demands[time_t] > 0:
    #             failed to distribute
                return False, -1, None, None
#     successfully distribute
    cost = sum([s.get_cost(time_t) for s in servers_order])
    return True, cost, servers_order, clients

if __name__ == '__main__':

    pass


