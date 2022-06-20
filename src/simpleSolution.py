from util import *
import numpy as np


def simple():
    """ 简单的贪心分配策略 """
    clients, demand_data = read_demand()
    qos_constraint = read_config()
    servers, server_bandwidths, server2index = read_bandwidth()
    qos_data = read_qos(server2index, len(servers), len(clients))

    time_num, client_num = demand_data.shape
    server_num = len(servers)

    output = []
    for t in range(time_num):
        bandwidth_remain = server_bandwidths.copy()

        for i in range(client_num):
            distribution = [clients[i]+':']
            demand = demand_data[t, i]
            for j in range(server_num):
                if qos_data[j, i] < qos_constraint and bandwidth_remain[j] > 0:
                    x = min(demand, bandwidth_remain[j])
                    bandwidth_remain[j] -= x
                    demand -= x
                    distribution.append(f'<{servers[j]},{x}>')
                    if demand == 0:
                        output.append(distribution)
                        break
            if demand > 0:
                raise Exception('Wrong distribution!')
    # 输出结果到 solution.txt
    with open('../output/solution.txt', mode='w+') as f:
        for distribution in output:
            f.writelines(distribution[0])
            for i in range(1, len(distribution)):
                if i < len(distribution)-1:
                    f.writelines(distribution[i]+',')
                else:
                    f.writelines(distribution[i]+'\n')


if __name__ == '__main__':
    simple()