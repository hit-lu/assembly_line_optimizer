import math
import random

import matplotlib.pyplot as plt

from optimizer_common import *
from dataloader import *


def get_top_k_value(pop_val, k: int):
    res = []
    pop_val_cpy = copy.deepcopy(pop_val)
    pop_val_cpy.sort(reverse=True)

    for i in range(min(len(pop_val_cpy), k)):
        for j in range(len(pop_val)):
            if abs(pop_val_cpy[i] - pop_val[j]) < 1e-9 and j not in res:
                res.append(j)
                break
    return res


def swap_mutation(component_points, individual):
    offspring = individual.copy()

    idx, component_index = 0, random.randint(0, len(component_points) - 1)
    for points in component_points.values():
        if component_index == 0:
            index1 = random.randint(0, points + max_machine_index - 2)
            while True:
                index2 = random.randint(0, points + max_machine_index - 2)
                if index1 != index2 and offspring[idx + index1] != offspring[idx + index2]:
                    break
            offspring[idx + index1], offspring[idx + index2] = offspring[idx + index2], offspring[idx + index1]
            break

        component_index -= 1
        idx += (points + max_machine_index - 1)

    return offspring


def roulette_wheel_selection(pop_eval):
    # Roulette wheel
    random_val = np.random.random()
    for idx, val in enumerate(pop_eval):
        random_val -= val
        if random_val <= 0:
            return idx
    return len(pop_eval) - 1


def random_selective(data, possibility):        # 依概率选择随机数
    assert len(data) == len(possibility) and len(data) > 0

    sum_val = sum(possibility)
    possibility = [p / sum_val for p in possibility]

    random_val = random.random()
    for idx, val in enumerate(possibility):
        random_val -= val
        if random_val <= 0:
            break
    return data[idx]


def selective_initialization(component_points, population_size):
    population = []                                     # population initialization

    for _ in range(population_size):
        individual = []
        for points in component_points.values():
            if points == 0:
                continue
            avl_machine_num = random.randint(1, min(max_machine_index, points))  # 可用机器数

            selective_possibility = []
            for p in range(1, avl_machine_num + 1):
                selective_possibility.append(pow(2, avl_machine_num - p + 1))

            sel_machine_num = random_selective([p + 1 for p in range(avl_machine_num)], selective_possibility)  # 选择的机器数
            sel_machine_set = random.sample([p for p in range(avl_machine_num)], sel_machine_num)

            sel_machine_points = [1 for _ in range(sel_machine_num)]
            for p in range(sel_machine_num - 1):
                if points == sum(sel_machine_points):
                    break
                assign_points = random.randint(1, points - sum(sel_machine_points))
                sel_machine_points[p] += assign_points

            if sum(sel_machine_points) < points:
                sel_machine_points[-1] += (points - sum(sel_machine_points))

            # code component allocation into chromosome
            for p in range(max_machine_index):
                if p in sel_machine_set:
                    individual += [0 for _ in range(sel_machine_points[0])]
                    sel_machine_points.pop(0)
                individual.append(1)
            individual.pop(-1)

        population.append(individual)
    return population


def selective_crossover(mother, father, non_decelerating=True):
    assert len(mother) == len(father)

    offspring1, offspring2 = mother.copy(), father.copy()
    one_counter, feasible_cutline = 0, []
    for idx in range(len(mother) - 1):
        if mother[idx] == 1:
            one_counter += 1
        if father[idx] == 1:
            one_counter -= 1

        # first constraint: the total number of “1”s (the number of partitions) in the chromosome is unchanged
        if one_counter != 0 or idx == 0 or idx == len(mother) - 2:
            continue

        # the selected cutline should guarantee there are the same or a larger number unassigned machine
        # for each component type
        n_bro, n_new = 0, 0
        if mother[idx] and mother[idx + 1]:
            n_bro += 1
        if father[idx] and father[idx + 1]:
            n_bro += 1
        if mother[idx] and father[idx + 1]:
            n_new += 1
        if father[idx] and mother[idx + 1]:
            n_new += 1

        # non_decelerating or accelerating crossover
        if (non_decelerating and n_bro <= n_new) or n_bro < n_new:
            feasible_cutline.append(idx)

    if len(feasible_cutline) == 0:
        return offspring1, offspring2

    cutline_idx = feasible_cutline[random.randint(0, len(feasible_cutline) - 1)]
    offspring1, offspring2 = mother[:cutline_idx + 1] + father[cutline_idx + 1:], father[:cutline_idx + 1] + mother[
                                                                                                     cutline_idx + 1:]
    return offspring1, offspring2


def cal_individual_val(component_points, component_nozzle, individual):
    idx, objective_val = 0, [0]
    machine_component_points = [[] for _ in range(max_machine_index)]

    # decode the component allocation
    for points in component_points.values():
        component_gene = individual[idx: idx + points + max_machine_index - 1]
        machine_idx, component_counter = 0, 0
        for gene in component_gene:
            if gene:
                machine_component_points[machine_idx].append(component_counter)
                machine_idx += 1
                component_counter = 0
            else:
                component_counter += 1
        machine_component_points[-1].append(component_counter)
        idx += (points + max_machine_index - 1)

    for machine_idx in range(max_machine_index):
        nozzle_points = defaultdict(int)
        for idx, nozzle in component_nozzle.items():
            if component_points[idx] == 0:
                continue
            nozzle_points[nozzle] += machine_component_points[machine_idx][idx]

        machine_points = sum(machine_component_points[machine_idx])         # num of placement points
        if machine_points == 0:
            continue
        ul = math.ceil(len(nozzle_points) * 1.0 / max_head_index) - 1       # num of nozzle set

        # assignments of nozzles to heads
        wl = 0                                                          # num of workload
        total_heads = (1 + ul) * max_head_index - len(nozzle_points)
        nozzle_heads = defaultdict(int)
        for nozzle in nozzle_points.keys():
            nozzle_heads[nozzle] = math.floor(nozzle_points[nozzle] * 1.0 / machine_points * total_heads)
            nozzle_heads[nozzle] += 1

        total_heads = (1 + ul) * max_head_index
        for heads in nozzle_heads.values():
            total_heads -= heads

        for nozzle in nozzle_heads.keys():      # TODO：有利于减少周期的方法
            if total_heads == 0:
                break
            nozzle_heads[nozzle] += 1
            total_heads -= 1

        # averagely assign placements to heads
        heads_placement = []
        for nozzle in nozzle_heads.keys():
            points = math.floor(nozzle_points[nozzle] / nozzle_heads[nozzle])

            heads_placement += [[nozzle, points] for _ in range(nozzle_heads[nozzle])]
            nozzle_points[nozzle] -= (nozzle_heads[nozzle] * points)
            for idx in range(len(heads_placement) - 1, -1, -1):
                if nozzle_points[nozzle] <= 0:
                    break
                nozzle_points[nozzle] -= 1
                heads_placement[idx][1] += 1
        heads_placement = sorted(heads_placement, key=lambda x: x[1], reverse=True)

        # every max_head_index heads in the non-decreasing order are grouped together as nozzle set
        for idx in range(len(heads_placement) // max_head_index):
            wl += heads_placement[idx][1]
        objective_val.append(T_pp * machine_points + T_tr * wl + T_nc * ul)

    return max(objective_val), machine_component_points


@timer_wrapper
def optimizer(pcb_data, component_data):
    # basic parameter
    # crossover rate & mutation rate: 80% & 10%
    # population size: 200
    # the number of generation: 500
    crossover_rate, mutation_rate = 0.8, 0.1
    population_size, n_generations = 200, 500

    # the number of placement points and nozzle type of component
    component_points, component_nozzle = defaultdict(int), defaultdict(str)
    for data in pcb_data.iterrows():
        part_index = component_data[component_data['part'] == data[1]['part']].index.tolist()[0]
        nozzle = component_data.loc[part_index]['nz']

        component_points[part_index] += 1
        component_nozzle[part_index] = nozzle

    # population initialization
    best_popval = []
    population = selective_initialization(component_points, population_size)
    with tqdm(total=n_generations) as pbar:
        pbar.set_description('genetic process for PCB assembly')

        new_population, new_pop_val = [], []
        for _ in range(n_generations):
            # calculate fitness value
            pop_val = []
            for individual in population:
                val, _ = cal_individual_val(component_points, component_nozzle, individual)
                pop_val.append(val)

            best_popval.append(min(pop_val))
            # min-max convert
            max_val = max(pop_val)
            pop_val = list(map(lambda v: max_val - v, pop_val))

            sum_pop_val = sum(pop_val)
            pop_val = [v / sum_pop_val for v in pop_val]

            select_index = get_top_k_value(pop_val, population_size - len(new_pop_val))
            population = [population[idx] for idx in select_index]
            pop_val = [pop_val[idx] for idx in select_index]

            population += new_population
            for individual in new_population:
                val, _ = cal_individual_val(component_points, component_nozzle, individual)
                pop_val.append(val)

            # crossover and mutation
            new_population = []
            for pop in range(population_size):
                if pop % 2 == 0 and np.random.random() < crossover_rate:
                    index1 = roulette_wheel_selection(pop_val)
                    while True:
                        index2 = roulette_wheel_selection(pop_val)
                        if index1 != index2:
                            break

                    offspring1, offspring2 = selective_crossover(population[index1], population[index2])
                    if np.random.random() < mutation_rate:
                        offspring1 = swap_mutation(component_points, offspring1)

                    if np.random.random() < mutation_rate:
                        offspring1 = swap_mutation(component_points, offspring1)

                    new_population.append(offspring1)
                    new_population.append(offspring2)

            pbar.update(1)

    best_individual = population[np.argmin(pop_val)]
    val, result = cal_individual_val(component_points, component_nozzle, best_individual)
    print(result)

    plt.plot(best_popval)
    plt.show()
    # TODO: 计算实际的PCB整线组装时间


if __name__ == '__main__':
    # warnings.simplefilter('ignore')
    # 参数解析
    parser = argparse.ArgumentParser(description='assembly line optimizer implementation')
    parser.add_argument('--filename', default='PCB.txt', type=str, help='load pcb data')
    parser.add_argument('--auto_register', default=1, type=int, help='register the component according the pcb data')

    params = parser.parse_args()

    # 结果输出显示所有行和列
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # 加载PCB数据
    pcb_data, component_data, _ = load_data(params.filename, component_register=params.auto_register)  # 加载PCB数据

    optimizer(pcb_data, component_data)




