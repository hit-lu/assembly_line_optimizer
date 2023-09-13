import copy
import math
import random
import numpy as np

from base_optimizer.optimizer_common import *


def objective_value_calculate(component_assignment, component_nozzle, task_block_weight):
    machine_assembly_time = []
    for machine_index in range(max_machine_index):
        task_block_number, total_point_number = 0, sum(component_assignment[machine_index])
        nozzle_points, nozzle_heads = defaultdict(int), defaultdict(int)

        for part, points in enumerate(component_assignment[machine_index]):
            nozzle = component_nozzle[part]
            nozzle_points[nozzle] += points
            nozzle_heads[nozzle] = 1
        remaining_head = max_head_index - len(nozzle_heads)

        nozzle_fraction = []
        for nozzle, points in nozzle_points.items():
            val = remaining_head * points / total_point_number
            nozzle_heads[nozzle] += math.floor(val)
            nozzle_fraction.append([nozzle, val - math.floor(val)])

        remaining_head = max_head_index - sum(nozzle_heads.values())
        sorted(nozzle_fraction, key=lambda x: x[1])
        nozzle_fraction_index = 0
        while remaining_head > 0:
            nozzle_heads[nozzle_fraction[nozzle_fraction_index][0]] += 1
            remaining_head -= 1

        for nozzle, heads_number in nozzle_heads.items():
            task_block_number = max(task_block_weight, math.ceil(nozzle_points[nozzle] / heads_number))

        machine_assembly_time.append(
            (t_pick + t_place) * sum(component_assignment[machine_index]) + task_block_number * task_block_weight)
    return max(machine_assembly_time)


def random_component_assignment(component_points, component_nozzle, component_feeders, task_block_weight):
    component_points_cpy = copy.deepcopy(component_points)
    component_number = len(component_points_cpy)
    assignment_result = [[0 for _ in range(component_number)] for _ in range(max_machine_index)]

    # == the set of feasible component type for each nozzle type
    nozzle_part_list = defaultdict(list)
    for index, nozzle in enumerate(component_nozzle):
        nozzle_part_list[nozzle].append(index)
    # === ensure every nozzle types ===
    selected_part = []
    for part_list in nozzle_part_list.values():
        part = random.sample(part_list, 1)[0]
        machine_index = random.randint(0, max_machine_index - 1)

        assignment_result[machine_index][part] += 1
        component_points_cpy[part] -= 1
        selected_part.append(part)

    # === assign one placement which has not been selected ===
    for part in range(component_number):
        if part in selected_part:
            continue

        assignment_result[random.randint(0, max_machine_index - 1)][part] += 1
        component_points_cpy[part] -= 1

    machine_assign = list(range(max_machine_index))
    random.shuffle(machine_assign)
    finished_assign_counter = 0
    while finished_assign_counter < component_number:
        # todo: feeder limit restriction
        for machine_index in machine_assign:
            part = random.randint(0, component_number - 1)
            feeder_counter = 0
            for idx in range(max_machine_index):
                if assignment_result[idx][part] > 0 or idx == machine_index:
                    feeder_counter += 1

            if component_points_cpy[part] == 0 or feeder_counter > component_feeders[part]:
                continue

            points = random.randint(1, component_points_cpy[part])
            assignment_result[machine_index][part] += points
            component_points_cpy[part] -= points
            if component_points_cpy[part] == 0:
                finished_assign_counter += 1

    assert sum(component_points_cpy) == 0

    return objective_value_calculate(assignment_result, component_nozzle, task_block_weight), assignment_result


def greedy_component_assignment(component_points, component_nozzle, component_feeders, task_block_weight):
    pass    # 不清楚原文想说什么


def local_search_component_assignment(component_points, component_nozzle, component_feeders, task_block_weight):
    # maximum number of iterations : 5000
    # maximum number of unsuccessful iterations: 50
    component_number = len(component_points)
    iteration_counter, unsuccessful_iteration_counter = 5000, 50
    optimal_val, optimal_assignment = random_component_assignment(component_points, component_nozzle, component_feeders,
                                                                  task_block_weight)
    for _ in range(iteration_counter):
        machine_index = random.randint(0, max_machine_index - 1)
        if sum(optimal_assignment[machine_index]) == 0:
            continue

        part_set = []
        for component_index in range(component_number):
            if optimal_assignment[machine_index][component_index] != 0:
                part_set.append(component_index)
        component_index = random.sample(part_set, 1)[0]
        r = random.randint(1, optimal_assignment[machine_index][component_index])

        assignment = copy.deepcopy(optimal_assignment)
        cyclic_counter = 0
        swap_machine_index = None
        while cyclic_counter <= 2 * machine_index:
            cyclic_counter += 1
            swap_machine_index = random.randint(0, max_machine_index - 1)
            feeder_available = 0
            for machine in range(max_machine_index):
                if optimal_assignment[machine][component_index] or machine == swap_machine_index:
                    feeder_available += 1

            if feeder_available <= component_feeders[component_index] and swap_machine_index != machine_index:
                break
        assert swap_machine_index is not None
        assignment[machine_index][component_index] -= r
        assignment[swap_machine_index][component_index] += r
        val = objective_value_calculate(assignment, component_nozzle, task_block_weight)
        if val < optimal_val:
            optimal_assignment, optimal_val = assignment, val
            unsuccessful_iteration_counter = 50
        else:
            unsuccessful_iteration_counter -= 1
            if unsuccessful_iteration_counter <= 0:
                break

    return optimal_val, optimal_assignment


def reconfig_crossover_operation(component_points, component_feeders, parent1, parent2):
    offspring1, offspring2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    component_number = len(component_points)

    # === crossover ===
    mask_bit = []
    for _ in range(max_machine_index):
        mask_bit.append(random.randint(0, 1))
    if sum(mask_bit) == 0 or sum(mask_bit) == max_machine_index:
        return offspring1, offspring2

    for machine_index in range(max_machine_index):
        if mask_bit:
            offspring1[machine_index] = copy.deepcopy(parent1[machine_index])
            offspring2[machine_index] = copy.deepcopy(parent2[machine_index])
        else:
            offspring1[machine_index] = copy.deepcopy(parent2[machine_index])
            offspring2[machine_index] = copy.deepcopy(parent1[machine_index])

    # === balancing ===
    # equally to reach the correct number
    for component_index in range(component_number):
        for offspring in [offspring1, offspring2]:
            additional_points = sum([offspring[mt][component_index] for mt in range(max_machine_index)]) - \
                                component_points[component_index]
            if additional_points > 0:
                # if a component type has more placements, decrease the assigned values on every head equally keeping
                # the proportion of the number of placement among the heads
                points_list = []
                for machine_index in range(max_machine_index):
                    points = math.floor(
                        additional_points * offspring[machine_index][component_index] / component_points[component_index])
                    points_list.append(points)
                    offspring[machine_index][component_index] -= points
                additional_points -= sum(points_list)

                for machine_index in range(max_machine_index):
                    if additional_points == 0:
                        break
                    if offspring[machine_index][component_index] == 0:
                        continue
                    offspring[machine_index][component_index] -= 1
                    additional_points += 1
            elif additional_points < 0:
                # otherwise, increase the assigned nonzero values equally
                machine_set = []
                for machine_index in range(max_machine_index):
                    if offspring[machine_index][component_index] == 0:
                        continue
                    machine_set.append(machine_index)

                points = -math.ceil(additional_points / len(machine_set))
                for machine_index in machine_set:
                    offspring[machine_index][component_index] += points
                    additional_points += points

                for machine_index in machine_set:
                    if additional_points == 0:
                        break
                    offspring[machine_index][component_index] += 1
                    additional_points -= 1

    for part in range(component_number):
        pt = 0
        for mt in range(max_machine_index):
            pt+= offspring1[mt][part]
        if pt!=component_points[part]:
            print('')
    for part in range(component_number):
        pt = 0
        for mt in range(max_machine_index):
            pt+= offspring2[mt][part]
        if pt!=component_points[part]:
            print('')
    return offspring1, offspring2


def reconfig_mutation_operation(component_feeders, parent):
    offspring = copy.deepcopy(parent)

    swap_direction = random.randint(0, 1)
    if swap_direction:
        swap_machine1, swap_machine2 = random.sample(list(range(max_machine_index)), 2)
    else:
        swap_machine2, swap_machine1 = random.sample(list(range(max_machine_index)), 2)

    component_list = []
    for component_index, points in enumerate(offspring[swap_machine1]):
        if points:
            component_list.append(component_index)
    swap_component_index = random.sample(component_list, 1)[0]
    swap_points = random.randint(1, offspring[swap_machine1][swap_component_index])

    feeder_counter = 0
    for machine_index in range(max_machine_index):
        if offspring[swap_machine1][swap_component_index] < swap_points or machine_index == swap_machine2:
            feeder_counter += 1
    if feeder_counter > component_feeders[swap_component_index]:
        return offspring

    offspring[swap_machine1][swap_component_index] -= swap_points
    offspring[swap_machine2][swap_component_index] += swap_points
    return offspring


def evolutionary_component_assignment(component_points, component_nozzle, component_feeders, task_block_weight):
    # population size: 10
    # probability of the mutation: 0.1
    # probability of the crossover: 0.8
    # number of generation: 100
    population_size = 10
    generation_number = 100
    mutation_rate, crossover_rate = 0.1, 0.8

    population = []
    for _ in range(population_size):
        population.append(
            random_component_assignment(component_points, component_nozzle, component_feeders, task_block_weight)[1])

    with tqdm(total=generation_number) as pbar:
        pbar.set_description('evolutionary algorithm process for PCB assembly line balance')

        new_population = []
        for _ in range(generation_number):
            # calculate fitness value
            pop_val = []
            for individual in population:
                pop_val.append(objective_value_calculate(individual, component_nozzle, task_block_weight))

            select_index = get_top_k_value(pop_val, population_size - len(new_population), reverse=False)
            population = [population[idx] for idx in select_index]
            pop_val = [pop_val[idx] for idx in select_index]

            population += new_population
            for individual in new_population:
                pop_val.append(objective_value_calculate(individual, component_nozzle, task_block_weight))

            # min-max convert
            max_val = max(pop_val)
            pop_val_sel = list(map(lambda v: max_val - v, pop_val))
            sum_pop_val = sum(pop_val_sel) + 1e-10
            pop_val_sel = [v / sum_pop_val + 1e-3 for v in pop_val_sel]

            # crossover and mutation
            new_population = []
            for pop in range(population_size):
                if pop % 2 == 0 and np.random.random() < crossover_rate:
                    index1 = roulette_wheel_selection(pop_val_sel)
                    while True:
                        index2 = roulette_wheel_selection(pop_val_sel)
                        if index1 != index2:
                            break

                    offspring1, offspring2 = reconfig_crossover_operation(component_points, component_feeders,
                                                                          population[index1], population[index2])

                    if np.random.random() < mutation_rate:
                        offspring1 = reconfig_mutation_operation(component_feeders, offspring1)

                    if np.random.random() < mutation_rate:
                        offspring2 = reconfig_mutation_operation(component_feeders, offspring2)

                    new_population.append(offspring1)
                    new_population.append(offspring2)

            pbar.update(1)

    return min(pop_val), population[np.argmin(pop_val)]


def reconfiguration_optimizer(pcb_data, component_data):
    # === data preparation ===
    component_number = len(component_data)

    component_points = [0 for _ in range(component_number)]
    component_nozzle = [0 for _ in range(component_number)]
    component_feeders = [0 for _ in range(component_number)]
    component_part = [0 for _ in range(component_number)]

    for _, data in pcb_data.iterrows():
        part_index = component_data[component_data['part'] == data['part']].index.tolist()[0]
        nozzle = component_data.loc[part_index]['nz']

        component_points[part_index] += 1
        component_nozzle[part_index] = nozzle
        component_part[part_index] = data['part']

        component_feeders[part_index] = component_data.loc[part_index]['feeder-limit']

    # === assignment of heads to modules is omitted ===
    optimal_assignment, optimal_val = [], None

    task_block_weight = 5   # element from list [0, 1, 2, 5, 10]
    # === assignment of components to heads
    for i in range(4):
        if i == 0:
            val, assignment = random_component_assignment(component_points, component_nozzle, component_feeders,
                                                          task_block_weight)
        elif i == 1:
            continue
        elif i == 2:
            val, assignment = local_search_component_assignment(component_points, component_nozzle,
                                                                component_feeders, task_block_weight)
        else:
            val, assignment = evolutionary_component_assignment(component_points, component_nozzle,
                                                                component_feeders, task_block_weight)

        if optimal_val is None or val < optimal_val:
            optimal_val, optimal_assignment = val, assignment.copy()

    if optimal_val is None:
        raise Exception('no feasible solution! ')

    return optimal_assignment
