import copy
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def read_data(dir_path):
    with open(f'{dir_path}/config.ini', 'r') as file:
        qos_constraint = file.readlines()[1].strip().split('=')[1]
    qos_constraint = int(qos_constraint)

    customers = {}
    max_time_t = 0
    with open(f'{dir_path}/demand.csv', 'r') as file:
        for i, line in enumerate(file.readlines()):
            line = line.strip().split(',')[1:]
            if i == 0:
                customers_list = copy.deepcopy(line)
                for c in customers_list:
                    customers[c] = []
            else:
                assert len(customers_list) == len(line)
                max_time_t += 1
                for c, j in zip(customers_list, line):
                    customers[c].append(int(j))

    servers = {}
    with open(f'{dir_path}/site_bandwidth.csv', 'r') as file:
        for line in file.readlines()[1:]:
            line = line.strip().split(',')
            assert line[0] not in servers
            servers[line[0]] = int(line[1])

    server_customers_qos = {}
    with open(f'{dir_path}/qos.csv', 'r') as file:
        for i, line in enumerate(file.readlines()):
            line = line.strip().split(',')
            if i == 0:
                temp = copy.deepcopy(line[1:])
                for i in customers: assert i in temp
            else:
                assert line[0] not in server_customers_qos and line[0] in servers and len(temp) == len(line[1:])
                server_customers_qos[line[0]] = {c: int(i) for c, i in zip(temp, line[1:])}

    return max_time_t, qos_constraint, customers, servers, server_customers_qos

def write_solution(clients, path, max_time_t):
    """
    [time1: {customer: {s: band, }, }, time2.....]
    """
    with open(path, 'w') as file:
        for time_t in range(max_time_t):
            for c in clients.values():
                if len(c.history[time_t]) == 0:
                    file.write(f'{c.name}:')
                else:
                    distribution = c.history[time_t]
                    temp = []
                    for k, v in distribution.items():
                        if v > 0: temp.append(f"<{k},{v}>")
                    items = ','.join(temp)
                    file.write(f'{c.name}:{items}')
                file.write('\n')



if __name__ == '__main__':
    qos_constraint, customers, servers, server_customers_qos = read_data('./data')
    print(softmax([1, 2, 3]))
    pass