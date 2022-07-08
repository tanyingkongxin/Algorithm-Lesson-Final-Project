import random
import math
import copy
import numpy as np
import statistics
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
    def __init__(self, name: str, bandwidth: int, qos_constraint, qos_table, max_time_num, epsilon=0.5):
        self.name = name
        self.bandwidth = bandwidth
        self.qos_constraint = qos_constraint
        self.qos_table = qos_table
        self.history_bands = None
        self.reserve_bands = [self.bandwidth for _ in range(max_time_num)]
        self.history_bands = [{} for _ in range(max_time_num)]
        self.max_time_num = max_time_num
        self.balance_current_future_epsilon = epsilon

    def __repr__(self):
        return f"server_{self.name}"

    def __hash__(self):
        return hash(self.name)

    def reset(self, time_t, client_name):
        self.history_bands[time_t] = {k: v for k, v in self.history_bands[time_t].items() if k != client_name}
        self.reserve_bands[time_t] = self.bandwidth - sum(self.history_bands[time_t].values())

    def get_median(self, time_t):
        temp = [sum(item.values()) for item in self.history_bands[:time_t + 1] if len(item) > 0]
        if len(temp) == 0: return 0
        return math.ceil(statistics.median(temp))

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
            future = self.get_cost(self.max_time_num - 1)
            return -(self.balance_current_future_epsilon * current + (1 - self.balance_current_future_epsilon) * future)
            # return -(0.9 * current + 0.1 * self.bandwidth)

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

class Solver():
    def __init__(self, data_dir, lifting_interval, epsilon, order_keys, mode=None, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        self.allowed_keys = {'server', 'max_serve', 'max_bands', 'next_step_max_bands', '95_free_lunch', 'estimate_lookahead'}
        for item in order_keys:
            assert item in self.allowed_keys
        self.data_dir = data_dir
        self.lifting_interval = lifting_interval
        self.order_keys = order_keys
        self.max_time_num, qos_constraint, customers, servers, server_customers_qos = read_data(self.data_dir)
        self.clients = {k: Client(k, v) for k, v in customers.items()}
        self.servers = {k: Server(k, v, qos_constraint, server_customers_qos, self.max_time_num, epsilon) for k, v in servers.items()}
        self.distribute_func = self.greedy_distribute if mode == 'greedy' else distribute

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

                if time_t == self.max_time_num - 1 or (time_t + 1) % self.lifting_interval == 0:
                    if debug:
                        print(f'starting lifting {time_t}')
                    iter_servers, iter_clients = lifting(iter_servers, iter_clients, self.max_time_num - 1, debug=debug)

                expected_costs = [s.get_cost(s.max_time_num - 1) for s in iter_servers.values()]
                pbar.set_postfix({'time': time_t, 'current_cost': cost, 'expected_cost': sum(expected_costs)})
                pbar.update(1)
                if debug:
                    print(expected_costs)

        return iter_servers, iter_clients

    def greedy_distribute(self, servers_order: [Server], clients: {str: Client}, time_t):
        servers_order, clients = copy.deepcopy(servers_order), copy.deepcopy(clients)
        def find_available_servers(servers: [Server], c: Client, time_t):
            available = {}
            for s in servers:
                s: Server
                if s.afford(c, time_t):
                    available[s] = s.get_cost(time_t)
            return available

        max_fail = 100
        for c in clients.values():
            c: Client
            fail = 1
            while fail < max_fail:
                available = find_available_servers(servers_order, c, time_t)
                for ii, (s, value_95) in enumerate(available.items()):
                    if c.demands[time_t] == 0: break
                    if s.free_lunch(time_t) > 0:
                        threshold = c.demands[time_t] / (len(available) - ii) + 50 * fail
                    else:
                        threshold = max(math.ceil(s.get_median(time_t) + fail * 50), value_95)

                    # if s.free_lunch(time_t) > 0:
                    #     rcc = max(s.get_median(time_t), value_95, math.ceil(0.8 * c.demands[time_t]), threshold)
                    #     upper = min(math.ceil(0.9 * c.demands[time_t]), s.reserve_bands[time_t])
                    # else:
                    #     upper = min(s.reserve_bands[time_t], c.demands[time_t])
                    #     rcc = min(s.get_median(time_t), threshold)
                    upper = min(c.demands[time_t], s.reserve_bands[time_t])
                    # sp = random.randint(min(threshold, upper), upper)
                    sp = math.ceil(min(upper, threshold))
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
    clients = [(c, c.demands[time_t]) for c in clients.values()]
    clients = list(sorted(clients, key=lambda item: item[1], reverse=True))
    for c, _ in clients:
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
    clients = {c.name: c for c, _ in clients}
    cost = sum([s.get_cost(time_t) for s in servers_order])
    return True, cost, servers_order, clients


def lifting(servers: {str: Server}, clients: {str: Client}, time_t, debug=False):
    def transfer(c: Client, src_server: Server, tgt_server: Server, time_t, transfer_bands):
        assert transfer_bands <= src_server.history_bands[time_t][c.name]
        assert transfer_bands <= tgt_server.reserve_bands[time_t]
        assert transfer_bands > 0
        src_server.history_bands[time_t][c.name] -= transfer_bands
        src_server.reserve_bands[time_t] += transfer_bands

        if c.name not in tgt_server.history_bands[time_t]:
            tgt_server.history_bands[time_t][c.name] = 0
        tgt_server.history_bands[time_t][c.name] += transfer_bands
        tgt_server.reserve_bands[time_t] -= transfer_bands

        c.history[time_t][src_server.name] = src_server.history_bands[time_t][c.name]
        c.history[time_t][tgt_server.name] = tgt_server.history_bands[time_t][c.name]

    servers = [(s, s.get_cost(time_t)) for s in servers.values()]

    best_cost = -1
    now_cost = sum([item[0].get_cost(time_t) for item in servers])
    before_lifting = now_cost
    while True:
        # now_cost = sum([item[0].get_cost(time_t) for item in servers])
        if best_cost < 0 or now_cost < best_cost:
            backup = copy.deepcopy(servers), copy.deepcopy(clients)
            best_cost = now_cost
            if debug:
                print(f'now cost is {best_cost}, before lifting {before_lifting}')
        else:
            break
        servers = list(sorted(servers, key=lambda item: item[1], reverse=True))
        for si, (s, cost_95) in enumerate(servers):
            s: Server
            for idx in range(0, time_t + 1):
                now_cost = s.bandwidth - s.reserve_bands[idx]
                if now_cost >= cost_95:
                    for c_name, bands in s.history_bands[idx].items():
                        for ss, ss_cost95 in servers[si+1:]:
                            ss: Server
                            if s == ss: continue
                            if ss.afford(clients[c_name], idx):
                                tgt_cost_now = ss.bandwidth - ss.reserve_bands[idx]
                                if ss_cost95 == 0:
                                    pass
                                transfer_bands = min(ss.reserve_bands[idx], ss_cost95 - tgt_cost_now, bands)
                                # if ss_cost95 == 0:
                                    # print(transfer_bands, ss_cost95, tgt_cost_now, s.free_lunch(time_t))
                                if transfer_bands > 0:
                                    transfer(clients[c_name], s, ss, idx, transfer_bands)
                                    bands -= transfer_bands
                                    if bands == 0: break
                                # best_bands = 0
                                # best_temp_s_and_ss_cost = s.get_cost(time_t) + ss.get_cost(time_t)
                                # for temp_b in range(1, min(bands, ss.reserve_bands[idx]), 100):
                                #     transfer(clients[c_name], s, ss, idx, temp_b)
                                #     temp_s_and_ss_cost_after = s.get_cost(time_t) + ss.get_cost(time_t)
                                #     if temp_s_and_ss_cost_after < best_temp_s_and_ss_cost:
                                #         best_bands = temp_b
                                #         best_temp_s_and_ss_cost = temp_s_and_ss_cost_after
                                #     transfer(clients[c_name], ss, s, idx, temp_b)
                                #
                                # if best_bands > 0:
                                #     transfer(clients[c_name], s, ss, idx, best_bands)
                                #     bands -= best_bands


    servers, clients = backup
    servers = {item[0].name: item[0] for item in servers}

    return servers, clients


if __name__ == '__main__':

    pass


