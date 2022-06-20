from util import *
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import random
import math
# from test import runTest

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
            distribution = [clients[i] + ':']
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
                if i < len(distribution) - 1:
                    f.writelines(distribution[i] + ',')
                else:
                    f.writelines(distribution[i] + '\n')


def linearProgramming():
    """ 整数规划，参考知乎的方案，使用 gurobi 来求解，链接：...... """
    # read data
    clients, demand_data = read_demand()
    qos_constraint = read_config()
    servers, server_bandwidths, server2index = read_bandwidth()
    qos_data = read_qos(server2index, len(servers), len(clients))

    time_num, client_num = demand_data.shape
    server_num = len(servers)

    # set model variable and object
    model = gp.Model('flow')
    x = model.addVars(client_num, server_num, time_num, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='x')
    w = model.addVars(server_num, time_num, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='w')
    w_max = model.addVars(server_num, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='w_max')  # w_max:代表模型中的 s_hat
    model.setObjective(gp.quicksum(w_max[j] for j in range(server_num)), GRB.MINIMIZE)

    # set constraint
    model.addConstrs((x[i, j, t] == 0 for i in range(client_num) for j in range(server_num) for t in range(time_num)
                      if qos_data[j, i] >= qos_constraint), name='qos')
    model.addConstrs((gp.quicksum(x[i, j, t] for j in range(server_num)) == demand_data[t, i] for i in range(client_num)
                     for t in range(time_num)), name='demand')
    model.addConstrs((gp.quicksum(x[i, j, t] for i in range(client_num)) <= server_bandwidths[j] for j in range(server_num)
                      for t in range(time_num)), name='bandwidth_limit')
    model.addConstrs((gp.quicksum(x[i, j, t] for i in range(client_num)) == w[j, t] for j in range(server_num)
                      for t in range(time_num)), name="middle variable constraint")

    quantile_point = math.ceil(0.95 * time_num)
    sample_t = random.sample([i for i in range(time_num)], quantile_point)
    model.addConstrs((w_max[j] >= w[j, t] for j in range(server_num) for t in range(time_num) if t in sample_t),
                     name='95')

    # solve
    model.setParam('TimeLimit', 300)
    model.optimize()

    # output solution
    N = client_num * server_num * time_num
    x_result = np.array([v.x for v in model.getVars()[0:N]], dtype=int).reshape((client_num, server_num, time_num))
    print(x_result.shape)

    # write solution to file
    with open('../output/solution.txt', mode='w+') as f:
        for t in range(time_num):
            for i in range(client_num):
                f.writelines(clients[i] + ':')
                line = [(j, x_result[i, j, t]) for j in range(server_num) if x_result[i, j, t] > 0]
                for k in range(len(line)):
                    if k < len(line)-1:
                        f.writelines(f'<{servers[line[k][0]]},{line[k][1]}>,')
                    else:
                        f.writelines(f'<{servers[line[k][0]]},{line[k][1]}>\n')


def linearProgramming_improved():
    """ 引入新决策变量的整数规划 """
    # read data
    clients, demand_data = read_demand()
    qos_constraint = read_config()
    servers, server_bandwidths, server2index = read_bandwidth()
    qos_data = read_qos(server2index, len(servers), len(clients))

    time_num, client_num = demand_data.shape
    server_num = len(servers)

    # set model variable and object
    model = gp.Model('flow')
    x = model.addVars(client_num, server_num, time_num, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='x')
    w = model.addVars(server_num, time_num, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='w')
    w_max = model.addVars(server_num, lb=0, ub=GRB.INFINITY, vtype=GRB.INTEGER, name='w_max')  # w_max:代表模型中的 s_hat
    y = model.addVars(server_num, time_num, vtype=GRB.BINARY, name='y')
    model.setObjective(gp.quicksum(w_max[j] for j in range(server_num)), GRB.MINIMIZE)

    # set constraint
    model.addConstrs((x[i, j, t] == 0 for i in range(client_num) for j in range(server_num) for t in range(time_num)
                      if qos_data[j, i] >= qos_constraint), name='qos')
    model.addConstrs((gp.quicksum(x[i, j, t] for j in range(server_num)) == demand_data[t, i] for i in range(client_num)
                     for t in range(time_num)), name='demand')
    model.addConstrs((gp.quicksum(x[i, j, t] for i in range(client_num)) <= server_bandwidths[j] for j in range(server_num)
                      for t in range(time_num)), name='bandwidth_limit')
    model.addConstrs((gp.quicksum(x[i, j, t] for i in range(client_num)) == w[j, t] for j in range(server_num)
                      for t in range(time_num)), name="middle variable constraint")
    model.addConstrs((gp.quicksum(y[j, t] for t in range(time_num)) <= 0.05 * time_num for j in range(server_num)),
                     name='y_constraint')

    # quantile_point = math.ceil(0.95 * time_num)
    # sample_t = random.sample([i for i in range(time_num)], quantile_point)
    # model.addConstrs((w_max[j] >= w[j, t] for j in range(server_num) for t in range(time_num) if t in sample_t),
    #                  name='95')
    M = 999999
    model.addConstrs((w_max[j] >= w[j, t] - M * y[j, t] for j in range(server_num)
                      for t in range(time_num)), name='95')

    # solve
    model.setParam('TimeLimit', 300)
    model.optimize()

    # output solution
    N = client_num * server_num * time_num
    x_result = np.array([v.x for v in model.getVars()[0:N]], dtype=int).reshape((client_num, server_num, time_num))
    print(x_result.shape)

    # write solution to file
    with open('../output/solution.txt', mode='w+') as f:
        for t in range(time_num):
            for i in range(client_num):
                f.writelines(clients[i] + ':')
                line = [(j, x_result[i, j, t]) for j in range(server_num) if x_result[i, j, t] > 0]
                for k in range(len(line)):
                    if k < len(line)-1:
                        f.writelines(f'<{servers[line[k][0]]},{line[k][1]}>,')
                    else:
                        f.writelines(f'<{servers[line[k][0]]},{line[k][1]}>\n')


if __name__ == '__main__':
    # simple() # 177272
    # linearProgramming() # 186427
    linearProgramming_improved()
    # runTest()

