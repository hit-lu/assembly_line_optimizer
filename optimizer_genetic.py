# implementation of <<An integrated allocation method for the PCB assembly line balancing problem with nozzle changes>>
import copy

import matplotlib.pyplot as plt

from base_optimizer.optimizer_common import *


def selective_initialization(component_points, component_feeders, population_size):
    population = []    # population initialization
    for _ in range(population_size):
        individual = []
        for part_index, points in component_points:
            if points == 0:
                continue
            # 可用机器数
            avl_machine_num = random.randint(1, min(max_machine_index, component_feeders[part_index], points))

            selective_possibility = []
            for p in range(1, avl_machine_num + 1):
                selective_possibility.append(pow(2, avl_machine_num - p + 1))

            sel_machine_num = random_selective([p + 1 for p in range(avl_machine_num)], selective_possibility)  # 选择的机器数
            sel_machine_set = random.sample([p for p in range(max_machine_index)], sel_machine_num)

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


def selective_crossover(component_points, component_feeders, mother, father, non_decelerating=True):
    assert len(mother) == len(father)

    offspring1, offspring2 = mother.copy(), father.copy()
    one_counter, feasible_cut_line = 0, []

    idx = 0
    for part_index, points in component_points:
        one_counter = 0

        idx_, mother_cut_line, father_cut_line = 0, [-1], [-1]
        for idx_, gene in enumerate(mother[idx: idx + points + max_machine_index - 1]):
            if gene:
                mother_cut_line.append(idx_)
        mother_cut_line.append(idx_ + 1)

        for idx_, gene in enumerate(father[idx: idx + points + max_machine_index - 1]):
            if gene:
                father_cut_line.append(idx_)
        father_cut_line.append(idx_ + 1)

        for offset in range(points + max_machine_index - 1):
            if mother[idx + offset] == 1:
                one_counter += 1
            if father[idx + offset] == 1:
                one_counter -= 1

            # first constraint: the total number of '1's (the number of partitions) in the chromosome is unchanged
            if one_counter != 0 or offset == 0 or offset == points + max_machine_index - 2:
                continue

            # the selected cut-line should guarantee there are the same or a larger number unassigned machine
            # for each component type
            n_bro, n_new = 0, 0
            if mother[idx + offset] and mother[idx + offset + 1]:
                n_bro += 1
            if father[idx + offset] and father[idx + offset + 1]:
                n_bro += 1
            if mother[idx + offset] and father[idx + offset + 1]:
                n_new += 1
            if father[idx + offset] and mother[idx + offset + 1]:
                n_new += 1

            # second constraint: non_decelerating or accelerating crossover
            if n_new < n_bro or (n_new == n_bro and not non_decelerating):
                continue

            # third constraint (customized constraint):
            # no more than the maximum number of available machine for each component type
            new_mother_cut_line, new_father_cut_line = [], []
            for idx_ in range(max_machine_index + 1):
                if mother_cut_line[idx_] <= offset:
                    new_mother_cut_line.append(mother_cut_line[idx_])
                else:
                    new_father_cut_line.append(mother_cut_line[idx_])

                if father_cut_line[idx_] <= offset:
                    new_father_cut_line.append(father_cut_line[idx_])
                else:
                    new_mother_cut_line.append(father_cut_line[idx_])

            sorted(new_mother_cut_line, reverse=False)
            sorted(new_father_cut_line, reverse=False)
            n_mother_machine, n_father_machine = 0, 0

            for idx_ in range(max_machine_index):
                if new_mother_cut_line[idx_ + 1] - new_mother_cut_line[idx_]:
                    n_mother_machine += 1

                if new_father_cut_line[idx_ + 1] - new_father_cut_line[idx_]:
                    n_father_machine += 1

            if n_mother_machine > component_feeders[part_index] or n_father_machine > component_feeders[part_index]:
                continue

            feasible_cut_line.append(idx + offset)

        idx += (points + max_machine_index - 1)

    if len(feasible_cut_line) == 0:
        return offspring1, offspring2

    cut_line_idx = feasible_cut_line[random.randint(0, len(feasible_cut_line) - 1)]
    offspring1, offspring2 = mother[:cut_line_idx + 1] + father[cut_line_idx + 1:], father[:cut_line_idx + 1] + mother[
                                                                                                     cut_line_idx + 1:]
    return offspring1, offspring2


def cal_individual_val(component_points, component_nozzle, individual):
    idx, objective_val = 0, []
    machine_component_points = [[] for _ in range(max_machine_index)]
    nozzle_component_points = defaultdict(list)

    # decode the component allocation
    for comp_idx, points in component_points:
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

        nozzle_component_points[component_nozzle[comp_idx]] = [0] * len(component_points)   # 初始化元件-吸嘴点数列表

    for comp_idx, points in component_points:
        nozzle_component_points[component_nozzle[comp_idx]][comp_idx] = points

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
            if nozzle_points[nozzle] == 0:
                continue
            nozzle_heads[nozzle] = math.floor(nozzle_points[nozzle] * 1.0 / machine_points * total_heads)
            nozzle_heads[nozzle] += 1

        total_heads = (1 + ul) * max_head_index
        for heads in nozzle_heads.values():
            total_heads -= heads

        while True:
            nozzle = max(nozzle_heads, key=lambda x: nozzle_points[x] / nozzle_heads[x])
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

        # the number of pick-up operations
        # (under the assumption of the number of feeder available for each comp. type is equal 1)
        pl = 0
        heads_placement_points = [0 for _ in range(max_head_index)]
        while True:
            head_assign_point = []
            for head in range(max_head_index):
                if heads_placement_points[head] != 0 or heads_placement[head] == 0:
                    continue

                nozzle, points = heads_placement[head]
                max_comp_index = np.argmax(nozzle_component_points[nozzle])

                heads_placement_points[head] = min(points, nozzle_component_points[nozzle][max_comp_index])
                nozzle_component_points[nozzle][max_comp_index] -= heads_placement_points[head]

                head_assign_point.append(heads_placement_points[head])

            min_points_list = list(filter(lambda x: x > 0, heads_placement_points))
            if len(min_points_list) == 0 or len(head_assign_point) == 0:
                break

            pl += max(head_assign_point)

            for head in range(max_head_index):
                heads_placement[head][1] -= min(min_points_list)
                heads_placement_points[head] -= min(min_points_list)

        # every max_head_index heads in the non-decreasing order are grouped together as nozzle set
        for idx in range(len(heads_placement) // max_head_index):
            wl += heads_placement[idx][1]
        objective_val.append(T_pp * machine_points + T_tr * wl + T_nc * ul + T_pl * pl)
    return objective_val, machine_component_points


def assemblyline_optimizer_genetic(pcb_data, component_data):
    # basic parameter
    # crossover rate & mutation rate: 80% & 10%
    # population size: 200
    # the number of generation: 500
    crossover_rate, mutation_rate = 0.8, 0.1
    population_size, n_generations = 500, 500

    # the number of placement points, the number of available feeders, and nozzle type of component respectively
    component_points, component_feeders, component_nozzle = defaultdict(int), defaultdict(int), defaultdict(str)
    for data in pcb_data.iterrows():
        part_index = component_data[component_data['part'] == data[1]['part']].index.tolist()[0]
        nozzle = component_data.loc[part_index]['nz']

        component_points[part_index] += 1
        component_feeders[part_index] = component_data.loc[part_index]['feeder-limit']
        component_nozzle[part_index] = nozzle

    component_points = sorted(component_points.items(), key=lambda x: x[0])     # 决定染色体排列顺序

    # population initialization
    best_popval = []
    population = selective_initialization(component_points, component_feeders, population_size)
    with tqdm(total=n_generations) as pbar:
        pbar.set_description('genetic algorithm process for PCB assembly line balance')

        new_population = []
        for _ in range(n_generations):
            # calculate fitness value
            pop_val = []
            for individual in population:
                val, assigned_points = cal_individual_val(component_points, component_nozzle, individual)
                pop_val.append(max(val))

            best_popval.append(min(pop_val))
            select_index = get_top_k_value(pop_val, population_size - len(new_population), reverse=False)
            population = [population[idx] for idx in select_index]
            pop_val = [pop_val[idx] for idx in select_index]

            population += new_population
            for individual in new_population:
                val, _ = cal_individual_val(component_points, component_nozzle, individual)
                pop_val.append(max(val))

            # min-max convert
            max_val = max(pop_val)
            pop_val = list(map(lambda v: max_val - v, pop_val))
            sum_pop_val = sum(pop_val) + 1e-10
            pop_val = [v / sum_pop_val + 1e-3 for v in pop_val]

            # crossover and mutation
            new_population = []
            for pop in range(population_size):
                if pop % 2 == 0 and np.random.random() < crossover_rate:
                    index1 = roulette_wheel_selection(pop_val)
                    while True:
                        index2 = roulette_wheel_selection(pop_val)
                        if index1 != index2:
                            break

                    offspring1, offspring2 = selective_crossover(component_points, component_feeders,
                                                                 population[index1], population[index2])

                    if np.random.random() < mutation_rate:
                        offspring1 = constraint_swap_mutation(component_points, offspring1)

                    if np.random.random() < mutation_rate:
                        offspring2 = constraint_swap_mutation(component_points, offspring2)

                    new_population.append(offspring1)
                    new_population.append(offspring2)

            pbar.update(1)

    best_individual = population[np.argmax(pop_val)]
    _, assignment_result = cal_individual_val(component_points, component_nozzle, best_individual)

    # available feeder check
    for part_index, data in component_data.iterrows():
        feeder_limit = data['feeder-limit']
        for machine_index in range(max_machine_index):
            if assignment_result[machine_index][part_index]:
                feeder_limit -= 1
        assert feeder_limit >= 0

    return assignment_result
