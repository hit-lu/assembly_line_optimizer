import math
import numpy as np

from base_optimizer.optimizer_common import *


# TODO: consider with the PCB placement topology
def assembly_time_estimator(component_points, component_feeders, component_nozzle, assignment_points):
    # todo: how to deal with nozzle change
    n_cycle, n_nz_change, n_gang_pick = 0, 0, 0

    nozzle_heads, nozzle_points = defaultdict(int), defaultdict(int)
    for idx, points in enumerate(assignment_points):
        if points == 0:
            continue
        nozzle_points[component_nozzle[idx]] += points
        nozzle_heads[component_nozzle[idx]] = 1

    while sum(nozzle_heads.values()) != max_head_index:
        max_cycle_nozzle = None

        for nozzle, head_num in nozzle_heads.items():
            if max_cycle_nozzle is None or nozzle_points[nozzle] / head_num > nozzle_points[max_cycle_nozzle] / \
                    nozzle_heads[max_cycle_nozzle]:
                max_cycle_nozzle = nozzle

        assert max_cycle_nozzle is not None
        nozzle_heads[max_cycle_nozzle] += 1

    n_cycle = max(map(lambda x: math.ceil(nozzle_points[x[0]] / x[1]), nozzle_heads.items()))

    # calculate the number of simultaneous pickup
    head_index, nozzle_cycle = 0, [[] for _ in range(max_head_index)]
    for nozzle, heads in nozzle_heads.items():
        head_index_cpy, points = head_index, nozzle_points[nozzle]
        for _ in range(heads):
            nozzle_cycle[head_index].append([nozzle, points // heads])
            head_index += 1

        points %= heads
        while points:
            nozzle_cycle[head_index_cpy][1] += 1
            points -= 1
            head_index_cpy += 1

    # nozzle_cycle_index = [0 for _ in range(max_head_index)]
    return n_cycle, n_nz_change, n_gang_pick


def assemblyline_optimizer_heuristic(pcb_data, component_data):
    # the number of placement points, the number of available feeders, and nozzle type of component respectively
    component_number = len(component_data)

    component_points = [0 for _ in range(component_number)]
    component_feeders = [0 for _ in range(component_number)]
    component_nozzle = [0 for _ in range(component_number)]
    component_part = [0 for _ in range(component_number)]

    nozzle_points = defaultdict(int)        # the number of placements of nozzle

    for _, data in pcb_data.iterrows():
        part_index = component_data[component_data['part'] == data['part']].index.tolist()[0]
        nozzle = component_data.loc[part_index]['nz']

        component_points[part_index] += 1
        component_feeders[part_index] = component_data.loc[part_index]['feeder-limit']
        # component_feeders[part_index] = math.ceil(component_data.loc[part_index]['feeder-limit'] / max_feeder_limit)
        component_nozzle[part_index] = nozzle
        component_part[part_index] = data['part']

        nozzle_points[nozzle] += 1

    # first step: generate the initial solution with equalized workload
    assignment_result = [[0 for _ in range(len(component_points))] for _ in range(max_machine_index)]
    assignment_points = [0 for _ in range(max_machine_index)]

    weighted_points = list(
        map(lambda x: x[1] + 1e-5 * nozzle_points[component_nozzle[x[0]]], enumerate(component_points)))

    for part_index in np.argsort(weighted_points):
        if (total_points := component_points[part_index]) == 0:        # total placements for each component type
            continue
        machine_set = []

        # define the machine that assigning placement points (considering the feeder limitation)
        for machine_index in np.argsort(assignment_points):
            if len(machine_set) >= component_points[part_index] or len(machine_set) >= component_feeders[part_index]:
                break
            machine_set.append(machine_index)

        # Allocation of mounting points to available machines according to the principle of equality
        while total_points:
            assign_machine = list(filter(lambda x: assignment_points[x] == min(assignment_points), machine_set))

            if len(assign_machine) == len(machine_set):
                # averagely assign point to all available machines
                points = total_points // len(assign_machine)
                for machine_index in machine_set:
                    assignment_points[machine_index] += points
                    assignment_result[machine_index][part_index] += points

                total_points -= points * len(assign_machine)
                for machine_index in machine_set:
                    if total_points == 0:
                        break
                    assignment_points[machine_index] += 1
                    assignment_result[machine_index][part_index] += 1
                    total_points -= 1
            else:
                # assigning placements to make up for the gap between the least and the second least
                second_least_machine, second_least_machine_points = -1, max(assignment_points) + 1
                for idx in machine_set:
                    if assignment_points[idx] < second_least_machine_points and assignment_points[idx] != min(
                            assignment_points):
                        second_least_machine_points = assignment_points[idx]
                        second_least_machine = idx

                assert second_least_machine != -1

                if len(assign_machine) * (second_least_machine_points - min(assignment_points)) < total_points:
                    min_points = min(assignment_points)
                    total_points -= len(assign_machine) * (second_least_machine_points - min_points)
                    for machine_index in assign_machine:
                        assignment_points[machine_index] += (second_least_machine_points - min_points)
                        assignment_result[machine_index][part_index] += (
                                    second_least_machine_points - min_points)
                else:
                    points = total_points // len(assign_machine)
                    for machine_index in assign_machine:
                        assignment_points[machine_index] += points
                        assignment_result[machine_index][part_index] += points

                    total_points -= points * len(assign_machine)
                    for machine_index in assign_machine:
                        if total_points == 0:
                            break
                        assignment_points[machine_index] += 1
                        assignment_result[machine_index][part_index] += 1
                        total_points -= 1

    # todo: implementation

    # second step: estimate the assembly time for each machine
    # third step: adjust the assignment results to reduce maximal assembly time among all machines

    return assignment_result
