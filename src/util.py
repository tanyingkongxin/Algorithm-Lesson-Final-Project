# read data
# generate ...
#
# test requirement

import numpy as np

import datetime


# def _datestr2num(date_str) -> int:
#     s_date = datetime.datetime.strptime(date_str, '%Y-%m-%dT%H:%M').timestamp()
#     return int(s_date)
data_dir = '../data2'

def read_demand():
    """ 读取客户节点的需求

    :return:
        client: 客户节点名称列表
        data: (time_num × client_num)numpy 数组，每一行表示某个时刻各个客户节点的带宽需求。
    """
    with open(f'{data_dir}/demand.csv') as f:
        clients = f.readline()[:-1].split(',')[1:]
        data = np.loadtxt(f, dtype=int, delimiter=',', usecols=range(1, len(clients)+1), encoding='utf-8')
    return clients, data


def read_config():
    """ 读取 qos 约束值 """
    with open(f"{data_dir}/config.ini") as f:
        qos_constraint = f.read().splitlines()[1].split('=')[1]
        return int(qos_constraint)


def read_bandwidth():
    """ 读取边缘节点提供的带宽上限

    :return:
        servers: list, 边缘节点名称列表
        server_bandwidths: numpy, 边缘节点带宽上限
        server2index: dict, 将边缘节点名称映射为 id
    """
    with open(f'{data_dir}/site_bandwidth.csv') as f:
        data = f.read().splitlines()[1:]
    servers = [''] * len(data)
    server_bandwidths = np.zeros(shape=len(data), dtype=int)
    server2index = {}
    for i, line in enumerate(data):
        l_line = line.split(',')
        servers[i], server_bandwidths[i] = l_line[0], l_line[1]
        server2index[l_line[0]] = i
    return servers, server_bandwidths, server2index


def read_qos(server2index: dict, num_server: int, num_client: int):
    """ 读取客户节点和边缘节点之间的 qos 矩阵

    :param server2index: 将边缘节点名称映射为 id 的字典
    :param num_server: 边缘节点个数
    :param num_client: 客户节点个数
    :return:
        qos_data: (num_server × num_client)numpy 数组，客户节点和边缘节点之间的 qos 矩阵
    """
    qos_data = np.zeros((num_server, num_client), dtype=int)
    with open(f"{data_dir}/qos.csv") as f:
        data = f.read().splitlines()
    for line in data[1:]:
        l_line = line.split(',')
        qos_data[server2index[l_line[0]]] = l_line[1:]
    return qos_data


if __name__ == '__main__':
    clients, demand_data = read_demand()
    qos_constraint = read_config()
    servers, server_bandwidths, server2index = read_bandwidth()
    qos_data = read_qos(server2index, len(servers), len(clients))
