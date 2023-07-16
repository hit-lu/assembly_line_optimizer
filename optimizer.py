import copy
import math

import matplotlib.pyplot as plt
import pandas as pd

from base_optimizer.optimizer_aggregation import *
from base_optimizer.optimizer_scanbased import *
from base_optimizer.optimizer_celldivision import *
from base_optimizer.optimizer_hybridgenetic import *
from base_optimizer.optimizer_feederpriority import *

from dataloader import *

from optimizer_genetic import *
from optimizer_heuristic import *


def deviation(data):
    assert len(data) > 0
    average, variance = sum(data) / len(data), 0
    for v in data:
        variance += (v - average) ** 2
    return variance / len(data)


def optimizer(pcb_data, component_data, assembly_line_optimizer, single_machine_optimizer):
    # todo: 由于吸嘴更换更因素的存在，在处理PCB8数据时，遗传算法因在负载均衡过程中对这一因素进行了考虑，性能更优
    assignment_result = assemblyline_optimizer_heuristic(pcb_data, component_data)
    # assignment_result = assemblyline_optimizer_genetic(pcb_data, component_data)
    print(assignment_result)

    assignment_result_cpy = copy.deepcopy(assignment_result)
    placement_points, placement_time = [], []
    partial_pcb_data, partial_component_data = defaultdict(pd.DataFrame), defaultdict(pd.DataFrame)
    for machine_index in range(max_machine_index):
        partial_pcb_data[machine_index] = pd.DataFrame(columns=pcb_data.columns)
        partial_component_data[machine_index] = component_data.copy(deep=True)
        placement_points.append(sum(assignment_result[machine_index]))

    assert sum(placement_points) == len(pcb_data)

    # === averagely assign available feeder ===
    for part_index, data in component_data.iterrows():
        feeder_limit = data['feeder-limit']
        feeder_points = [assignment_result[machine_index][part_index] for machine_index in range(max_machine_index)]

        for machine_index in range(max_machine_index):
            if feeder_points[machine_index] == 0:
                continue

            arg_feeder = max(math.floor(feeder_points[machine_index] / sum(feeder_points) * data['feeder-limit']), 1)

            partial_component_data[machine_index].loc[part_index]['feeder-limit'] = arg_feeder
            feeder_limit -= arg_feeder

        for machine_index in range(max_machine_index):
            if feeder_limit <= 0:
                break

            if feeder_points[machine_index] == 0:
                continue
            partial_component_data[machine_index].loc[part_index]['feeder-limit'] += 1
            feeder_limit -= 1

        for machine_index in range(max_machine_index):
            if feeder_points[machine_index] > 0:
                assert partial_component_data[machine_index].loc[part_index]['feeder-limit'] > 0

    # === assign placements ===
    component_machine_index = [0 for _ in range(len(component_data))]
    for _, data in pcb_data.iterrows():
        part_index = component_data[component_data['part'] == data['part']].index.tolist()[0]
        while True:
            machine_index = component_machine_index[part_index]
            if assignment_result[machine_index][part_index] == 0:
                component_machine_index[part_index] += 1
                machine_index += 1
            else:
                break
        assignment_result[machine_index][part_index] -= 1
        partial_pcb_data[machine_index] = pd.concat([partial_pcb_data[machine_index], pd.DataFrame(data).T])

    # === adjust the number of available feeders for single optimization separately ===
    for machine_index, data in partial_pcb_data.items():
        data = data.reset_index(drop=True)
        if len(data) == 0:
            continue

        part_info = []  # part info list：(part index, part points, available feeder-num, upper feeder-num)
        for part_index, cp_data in partial_component_data[machine_index].iterrows():
            if assignment_result_cpy[machine_index][part_index]:
                part_info.append(
                    [part_index, assignment_result_cpy[machine_index][part_index], 1, cp_data['feeder-limit']])

        part_info = sorted(part_info, key=lambda x: x[1], reverse=True)
        start_index, end_index = 0, min(max_head_index - 1, len(part_info) - 1)
        while start_index < len(part_info):
            assign_part_point, assign_part_index = [], []
            for idx_ in range(start_index, end_index + 1):
                for _ in range(part_info[idx_][2]):
                    assign_part_point.append(part_info[idx_][1] / part_info[idx_][2])
                    assign_part_index.append(idx_)

            variance = deviation(assign_part_point)
            while start_index != end_index:
                part_info_index = assign_part_index[np.argmax(assign_part_point)]

                if part_info[part_info_index][2] < part_info[part_info_index][3]:   # 供料器数目上限的限制
                    part_info[part_info_index][2] += 1
                    end_index -= 1

                    new_assign_part_point, new_assign_part_index = [], []
                    for idx_ in range(start_index, end_index + 1):
                        for _ in range(part_info[idx_][2]):
                            new_assign_part_point.append(part_info[idx_][1] / part_info[idx_][2])
                            new_assign_part_index.append(idx_)

                    new_variance = deviation(new_assign_part_point)
                    if variance < new_variance:
                        part_info[part_info_index][2] -= 1
                        end_index += 1
                        break

                    variance = new_variance
                    assign_part_index, assign_part_point = new_assign_part_index, new_assign_part_point
                else:
                    break

            start_index = end_index + 1
            end_index = min(start_index + max_head_index - 1, len(part_info) - 1)

        # update available feeder number
        max_avl_feeder = max(part_info, key=lambda x: x[2])[2]
        for info in part_info:
            partial_component_data[machine_index].loc[info[0]]['feeder-limit'] = math.ceil(info[2] / max_avl_feeder)

        placement_time.append(base_optimizer(machine_index + 1, data, partial_component_data[machine_index],
                                             feeder_data=pd.DataFrame(columns=['slot', 'part', 'arg']),
                                             method=single_machine_optimizer, hinter=True))

    average_time, standard_deviation_time = sum(placement_time) / max_machine_index, 0
    for machine_index in range(max_machine_index):
        print('assembly time for machine ' + str(machine_index + 1) + ': ' + str(
            placement_time[machine_index]) + ' s, ' + 'total placements: ' + str(placement_points[machine_index]))
        standard_deviation_time += pow(placement_time[machine_index] - average_time, 2)
    standard_deviation_time /= max_machine_index
    standard_deviation_time = math.sqrt(standard_deviation_time)

    print('finial assembly time: ' + str(max(placement_time)) + 's, standard deviation: ' + str(standard_deviation_time))


# todo: 不同类型元件的组装时间差异
def base_optimizer(machine_index, pcb_data, component_data, feeder_data=None, method='', hinter=False):

    if method == 'cell_division':  # 基于元胞分裂的遗传算法
        component_result, cycle_result, feeder_slot_result = optimizer_celldivision(pcb_data, component_data,
                                                                                    hinter=False)
        placement_result, head_sequence = greedy_placement_route_generation(component_data, pcb_data, component_result,
                                                                            cycle_result, feeder_slot_result)
    elif method == 'feeder_scan':  # 基于基座扫描的供料器优先算法
        # 第1步：分配供料器位置
        nozzle_pattern = feeder_allocate(component_data, pcb_data, feeder_data, figure=False)
        # 第2步：扫描供料器基座，确定元件拾取的先后顺序
        component_result, cycle_result, feeder_slot_result = feeder_base_scan(component_data, pcb_data, feeder_data,
                                                                              nozzle_pattern)

        # 第3步：贴装路径规划
        placement_result, head_sequence = greedy_placement_route_generation(component_data, pcb_data, component_result,
                                                                            cycle_result, feeder_slot_result)
        # placement_result, head_sequence = beam_search_for_route_generation(component_data, pcb_data, component_result,
        #                                                                    cycle_result, feeder_slot_result)

    elif method == 'hybrid_genetic':  # 基于拾取组的混合遗传算法
        component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = optimizer_hybrid_genetic(
            pcb_data, component_data, hinter=False)

    elif method == 'aggregation':  # 基于batch-level的整数规划 + 启发式算法
        component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = optimizer_aggregation(
            component_data, pcb_data)
    elif method == 'genetic_scanning':
        component_result, cycle_result, feeder_slot_result, placement_result, head_sequence = optimizer_genetic_scanning(
            component_data, pcb_data, hinter=False)
    else:
        raise 'method is not existed'

    if hinter:
        optimization_assign_result(component_data, pcb_data, component_result, cycle_result, feeder_slot_result,
                                   nozzle_hinter=True, component_hinter=False, feeder_hinter=False)

        print('----- Placement machine ' + str(machine_index) + ' ----- ')
        print('-Cycle counter: {}'.format(sum(cycle_result)))

        total_nozzle_change_counter, total_pick_counter = 0, 0
        total_pick_movement = 0
        assigned_nozzle = ['' if idx == -1 else component_data.loc[idx]['nz'] for idx in component_result[0]]

        for cycle in range(len(cycle_result)):
            pick_slot = set()
            for head in range(max_head_index):
                if (idx := component_result[cycle][head]) == -1:
                    continue

                nozzle = component_data.loc[idx]['nz']
                if nozzle != assigned_nozzle[head]:
                    if assigned_nozzle[head] != '':
                        total_nozzle_change_counter += 1
                    assigned_nozzle[head] = nozzle

                pick_slot.add(feeder_slot_result[cycle][head] - head * interval_ratio)
            total_pick_counter += len(pick_slot) * cycle_result[cycle]

            pick_slot = list(pick_slot)
            pick_slot.sort()
            for idx in range(len(pick_slot) - 1):
                total_pick_movement += abs(pick_slot[idx+1] - pick_slot[idx])

        print('-Nozzle change counter: {}'.format(total_nozzle_change_counter))
        print('-Pick operation counter: {}'.format(total_pick_counter))
        print('-Pick movement: {}'.format(total_pick_movement))
        print('------------------------------ ')

    # 估算贴装用时
    return placement_time_estimate(component_data, pcb_data, component_result, cycle_result, feeder_slot_result,
                                   placement_result, head_sequence, hinter=False)


@timer_wrapper
def main():
    # warnings.simplefilter('ignore')
    # 参数解析
    parser = argparse.ArgumentParser(description='assembly line optimizer implementation')
    parser.add_argument('--filename', default='PCB.txt', type=str, help='load pcb data')
    parser.add_argument('--auto_register', default=1, type=int, help='register the component according the pcb data')
    parser.add_argument('--base_optimizer', default='feeder_scan', type=str, help='base optimizer for single machine')
    parser.add_argument('--assembly_optimizer', default='heuristic', type=str, help='optimizer for PCB Assembly Line')
    parser.add_argument('--feeder_limit', default=3, type=int,
                        help='the upper feeder limit for each type of component')
    params = parser.parse_args()

    # 结果输出显示所有行和列
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # 加载PCB数据
    pcb_data, component_data, _ = load_data(params.filename, default_feeder_limit=params.feeder_limit,
                                            cp_auto_register=params.auto_register)  # 加载PCB数据

    optimizer(pcb_data, component_data, params.assembly_optimizer, params.base_optimizer)


if __name__ == '__main__':
    main()


