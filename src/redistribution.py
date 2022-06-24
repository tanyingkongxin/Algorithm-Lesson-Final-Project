'''
思路1：将边缘节点大于95分位时刻的流量分配给其他边缘节点；
思路2：干脆不管top 5的流量，将剩余95%的流量在边缘节点之间尽量做到均匀分布；
'''
import numpy as np
import timeit


class Distribution:

    def __init__(self, clients: list, servers: list, n_time: int, qos_data: np.ndarray, qos_constraint: int):
        self.clients = clients
        self.servers = servers
        self.n_time = n_time
        self.n_client = len(clients)
        self.n_server = len(servers)
        # self.data = [[[] for _ in range(len(clients))] for _ in range(n_time)]  # shape= n_time * n_client
        self.x = np.zeros(shape=(self.n_client, self.n_server, self.n_time), dtype=int)
        self.w = np.zeros(shape=(self.n_server, self.n_time), dtype=int)  # 不同时刻每个边缘节点分配的带宽情况

        self.top_num = int(0.05 * n_time)

        self.qos = qos_data < qos_constraint

        self.index95 = None

    def add(self, t, c, s, bandwidth):
        self.x[c, s, t] = bandwidth
        self.w[s, t] += bandwidth

    def save(self):
        # 输出结果到 solution.txt
        with open('../output/solution.txt', mode='w+') as f:
            for t in range(self.n_time):
                for i in range(self.n_client):
                    f.writelines(self.clients[i] + ':')
                    tmp = ','.join([
                        f'<{self.servers[j]},{self.x[i, j, t]}>' for j in range(self.n_server) if self.x[i, j, t] > 0])
                    f.writelines(tmp + '\n')

    def save_sol(self, y=None):
        if y is None:
            index_sorted = np.argsort(self.w, axis=1)
            y = np.zeros(shape=self.w.shape, dtype=bool)
            for s in range(self.n_server):
                for t in range(self.n_time):
                    if index_sorted[s, t] > self.n_time - self.top_num - 1:
                        y[s, t] = 1

        with open('../output/output.sol', mode='w+') as f:
            for c, s, t in np.ndindex(self.x.shape):
                f.writelines(f'x[{c},{s},{t}] {self.x[c,s,t]}\n')
            for s, t in np.ndindex(self.w.shape):
                f.writelines(f'w[{s},{t}] {self.w[s, t]}\n')
            for s, t in np.ndindex(y.shape):
                f.writelines(f'y[{s},{t}] {int(y[s, t])}\n')

        print('write sol file successfully.')

    def load(self, path):
        """ 从 solution.txt 载入分配方案 """
        server2index = dict(zip(self.servers, np.arange(self.n_server)))
        with open(path, 'r') as f:
            for t in range(self.n_time):
                for c in range(self.n_client):
                    line = f.readline()[:-1]  # 去除'\n'
                    i = line.find(':')
                    if i == len(line) - 1:  # 说明没有分配
                        continue
                    for item in line[i + 2:-1].split('>,<'):
                        server_name, bandwidth = item.split(',')
                        self.add(t, c, server2index[server_name], int(bandwidth))
        return None

    def get_index95(self):
        if self.index95 is None:
            index_sorted = np.argsort(self.w, axis=1)
            self.index95 = []
            for s in range(self.n_server):
                for t in range(self.n_time):
                    if index_sorted[s, t] == self.n_time - self.top_num - 1:
                        self.index95.append(t)
                        break
        return self.index95

    def update(self, s, t):
        c_list = [c for c in range(self.n_client) if self.x[c, s, t] > 0]
        index95 = self.get_index95()
        for i in np.arange(0, s) + np.arange(s+1, self.n_server):
            for c in c_list:
                if self.qos[i, c]:
                    if self.w[i, t] > self.w[i, index95[i]]: # top 5 时随便发送带宽
                        # update server s(d, x)
                        # update server i(d, x)
                        pass
                    else:
                        delta = min(self.w[i, index95[i]] - self.w[i, t], self.x[c, s, t])
                        self.add(t, c, s, -delta)
                        self.add(t, c, i, delta)
                        # update 95 percent cost of server_i

    def get_cost(self):
        index95 = self.get_index95()
        return np.sum(self.w[np.arange(self.n_server), index95])


# def redistribute(d: Distribution):
#     """
#     :param:
#         start: Distribution, 初始化的分配方案
#         qos_data: np.array, server_num × client_num
#
#     :return:
#         cost: int, 最终重分配方案的带宽成本
#     """
#     init_cost = d.get_cost()
#     iter_time = 1
#     while iter_time:
#         iter_time -= 1
#         for s in range(d.n_server):
#
#     # 记录 95 位置的 index，然后选择合适的兄弟节点，迁移带宽
#
#     pass


if __name__ == '__main__':
    pass
