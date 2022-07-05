import random
import math
import copy
import numpy as np

from utils import softmax


class Client():
    def __init__(self, name: str, demands: [int]):
        self.name = name
        self.demands = demands
        self.history = [{} for _ in self.demands]

    def update(self, server_name, time_t, bands):
        assert bands <= self.demands[time_t]
        self.demands[time_t] -= bands
        self.history[time_t][server_name] = bands

    def __repr__(self):
        return self.name

class Server():
    def __init__(self, name: str, bandwidth: int, qos_constraint, qos_table, max_time):
        self.name = name
        self.bandwidth = bandwidth
        self.qos_constraint = qos_constraint
        self.qos_table = qos_table
        self.history_bands = None
        self.reserve_bands = None
        self.history_bands = [{} for _ in range(max_time)]
        self.max_time_t = max_time

    def __repr__(self):
        return f"server_{self.name}"

    def afford(self, c: Client, time_t):
        if self.reserve_bands is None:
            self.reserve_bands = [self.bandwidth for _ in c.demands]
        return self.qos_table[self.name][c.name] < self.qos_constraint and self.reserve_bands[time_t] >= 0

    def execute(self, c: Client, time_t):
        assert self.afford(c, time_t)
        bands = min(self.reserve_bands[time_t], c.demands[time_t])
        self.history_bands[time_t][c.name] = bands
        self.reserve_bands[time_t] -= bands
        c.update(self.name, time_t, bands)

    def get_cost(self, end_time):
        costs = [sum(item.values()) for item in self.history_bands[:end_time + 1]]
        return sorted(costs)[math.ceil(len(costs) * 0.95) - 1]

    def free_lunch(self, time_t):
        costs = list(sorted([sum(item.values()) for item in self.history_bands[:time_t + 1]]))
        now_cost = costs[math.ceil(len(costs) * 0.95) - 1]
        lookahead = min(int(time_t / self.max_time_t * time_t), len(self.history_bands) - time_t - 1)
        if time_t < int(0.5 * self.max_time_t): lookahead = int(lookahead * 1.15)
        p  = max(time_t / self.max_time_t - 0.3, 0.0) + 0.2
        estimate_band = p * costs[-1] + (1 - p) * self.bandwidth
        costs += [estimate_band] * lookahead
        next_cost = costs[math.ceil(len(costs) * 0.95) - 1]
        return now_cost - next_cost


def get_server_order(clients: {str: Client}, servers: {str: Server}, time_t, use_history=False):
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
        servers_order.append((s, cnt, s.bandwidth, rbands, s.free_lunch(time_t)))

    key = lambda item: (item[1], item[2])
    if use_history: key = lambda item: (item[4], item[1], item[2])
    servers_order = list(sorted(servers_order, key=key, reverse=True))
    pass
    # print(1)
    return [item[0] for item in servers_order]

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


def sample_server_order(clients: {str: Client}, servers: {str: Server}, time_t, num):
    servers_order = get_server_order(clients, servers, time_t, use_history=True)
    p = 100
    weights = []
    for i, _ in enumerate(servers_order):
        weights.append(p)
        if i <= int(len(servers_order) / 2):
            p *= 0.9
        else:
            p *= 0.6
    populations = []
    weights = softmax(weights)
    while len(populations) < num:
        temp = np.random.choice(servers_order, len(servers_order), replace=False, p=weights).tolist()
        flag, cost, server_order_time_t, clients_time_t  = distribute(temp, clients, time_t)
        if flag:
            # temp already distribute time_t bands
            populations.append({time_t: [server_order_time_t, clients_time_t, cost]})
            print(f"cost in time{time_t} is {cost}")

    return populations

def calculate_cost(time_start, time_end, servers: {str: Server}, clients: {str: Client}):
    for time_t in range(time_start, time_end+1):
        if time_t == time_start:
            server_order_time_t, clients_time_t, cost = sample_server_order(clients, servers, time_t, 1)[0][time_t]
            iter_servers = {server.name: server for server in server_order_time_t}
            iter_clients = clients_time_t
        else:
            server_order_time_t, clients_time_t, cost = sample_server_order(iter_clients, iter_servers, time_t, 1)[0][time_t]
            iter_servers = {server.name: server for server in server_order_time_t}
            iter_clients = clients_time_t
        print(f"time_{time_t}, cost is {cost}")

    return iter_servers, iter_clients


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    dir = './data'
    from utils import read_data, write_solution

    max_time_t, qos_constraint, customers, servers, server_customers_qos = read_data(dir)
    clients = {k: Client(k, v) for k, v in customers.items()}
    servers = {k: Server(k, v, qos_constraint, server_customers_qos, max_time_t) for k, v in servers.items()}

    # solutions = sample_server_order(clients, servers, 0, 1)
    iter_servers, iter_clients = calculate_cost(0, max_time_t - 1, servers, clients)
    write_solution(iter_clients, './test.txt', max_time_t)


    pass


